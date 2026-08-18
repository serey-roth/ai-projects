
import asyncio
import re
from pathlib import Path
from typing import Optional, override

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer

from llama_index.core.node_parser import MarkdownNodeParser, SemanticDoubleMergingSplitterNodeParser, SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# We need to keep the intro to lists and code blocks togther
_INTRO_LINE_RE = r"(?:[^\n]*:[ \t]*\n\n?)?"
_CODE_FENCE_RE = _INTRO_LINE_RE + r"```.*?```"
_IMAGE_CAPTION_RE = r"!\[[^\]]*\]\([^)]*\)\n\n\*\*Figure \d+:.*?\*\*"
_TABLE_RE = r"(?:^\|.*\|[ \t]*\n?)+"
_LIST_RE = _INTRO_LINE_RE + r"(?:^(?:[-*]|\d+\.)[ \t]+.*\n?)+"

_ATOMIC_BLOCK_RE = re.compile(
    f"(?:{_CODE_FENCE_RE})|(?:{_IMAGE_CAPTION_RE})|(?:{_TABLE_RE})|(?:{_LIST_RE})",
    re.DOTALL | re.MULTILINE,
)

def block_aware_sentence_splitter(text: str) -> list[str]:
    default_split = split_by_sentence_tokenizer()
    units: list[str] = []
    pos = 0

    for match in _ATOMIC_BLOCK_RE.finditer(text):
        start, end = match.span()
        if start > pos:
            units.extend(default_split(text[pos:start]))
        units.append(match.group(0))
        pos = end

    if pos < len(text):
        units.extend(default_split(text[pos:]))

    return units


def _restore_block_spacing(text: str, atomic_blocks: list[str]) -> str:
    """
    SemanticDoubleMergingSplitterNodeParser strips every unit our sentence_splitter
    returns and rejoins them with a single space (merging_separator), regardless of
    what kind of unit it is - so the blank line between a heading/sentence and a
    following code/image/table block collapses to one space. 
    """
    for block in atomic_blocks:
        if block in text:
            text = text.replace(block, f"\n\n{block}\n\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_MIN_CHUNK_CHARS = 40
_HEADING_LINE_RE = re.compile(r"^#{1,6}\s")


def _ends_with_heading(text: str) -> bool:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return bool(lines) and bool(_HEADING_LINE_RE.match(lines[-1]))


def _starts_with_heading(text: str) -> bool:
    return bool(_HEADING_LINE_RE.match(text.strip()))


def _join_chunk_texts(a: str, b: str) -> str:
    # A single-space join reads fine for stray prose fragments, but a heading
    # must stay at the start of its own line to remain a heading at all - glue
    # "###### Patterns" onto "# Scaling Writes" with a space and you get
    # "###### Patterns # Scaling Writes", which is no longer a valid heading
    # if this text is ever re-parsed as markdown.
    separator = "\n\n" if _ends_with_heading(a) or _starts_with_heading(b) else " "
    return f"{a}{separator}{b}".strip()


def _merge_small_nodes(sub_nodes: list) -> list:
    """
    The semantic splitter can isolate a short text as its
    own chunk if it's dissimilar enough from its neighbors, making it a poor embedding.
    Fold anything under _MIN_CHUNK_CHARS into a neighbor.
    Special case is a heading, we want to merge it forward into its own body 
    instead of a tail of another chunk.
    """
    if len(sub_nodes) <= 1:
        return sub_nodes

    merged: list = []
    pending_prefix: Optional[str] = None

    for sub_node in sub_nodes:
        if pending_prefix is not None:
            sub_node.text = _join_chunk_texts(pending_prefix, sub_node.text)
            pending_prefix = None

        text = sub_node.text.strip()
        if len(text) >= _MIN_CHUNK_CHARS:
            merged.append(sub_node)
            continue

        if merged and not _starts_with_heading(text):
            merged[-1].text = _join_chunk_texts(merged[-1].text, sub_node.text)
        else:
            pending_prefix = sub_node.text

    if pending_prefix is not None:
        if merged:
            merged[-1].text = _join_chunk_texts(merged[-1].text, pending_prefix)
        else:
            return sub_nodes[-1:]

    return merged


class BlockAwareSemanticDoubleMergingSplitterNodeParser(SemanticDoubleMergingSplitterNodeParser):
    @override
    def build_semantic_nodes_from_nodes(self, nodes):
        """
        SemanticDoubleMergingSplitterNodeParser.build_semantic_nodes_from_nodes calls
        build_nodes_from_splits, which never copies source-node metadata onto the new
        sub-nodes (unlike MarkdownNodeParser, which does this as an explicit extra
        step). Also no block spacing are preserved for atomic elements.
        """
        all_nodes = []
        for node in nodes:
            original_text = node.get_content()
            atomic_blocks = [m.group(0).strip() for m in _ATOMIC_BLOCK_RE.finditer(original_text)]
    
            sub_nodes = super().build_semantic_nodes_from_nodes([node])
            for sub_node in sub_nodes:
                sub_node.metadata.update(node.metadata)
                sub_node.text = _restore_block_spacing(sub_node.text, atomic_blocks)
            all_nodes.extend(sub_nodes)
    
        return _merge_small_nodes(all_nodes)
    

if __name__ == "__main__":
    DATA_DIR = Path(__file__).resolve().parent / "data"

    MD_FILE_PATH = DATA_DIR / "how-to-scale-writes.md"

    assert MD_FILE_PATH.exists()

    print(f"Loading markdown documents from {DATA_DIR}...")
    reader = SimpleDirectoryReader(
        input_files=[str(MD_FILE_PATH)],
        required_exts=[".md"],
        filename_as_id=True,
        exclude_empty=True,
        exclude_hidden=True,
    )

    documents = asyncio.run(reader.aload_data(show_progress=False))
    parser = MarkdownNodeParser(
        include_metadata=True,
        include_prev_next_rel=True,
    )
               
    nodes = parser.get_nodes_from_documents(documents=documents, show_progress=True)

    splitter = BlockAwareSemanticDoubleMergingSplitterNodeParser(
        include_metadata=True,
        include_prev_next_rel=True,
        embed_model=HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            device="cpu",
        ),
        sentence_splitter=block_aware_sentence_splitter,
        max_chunk_size=1600,
        initial_threshold=0.5,
        appending_threshold=0.6,
        merging_threshold=0.6
    )

    semantic_nodes = splitter.build_semantic_nodes_from_nodes(nodes=nodes)
    print(f"{len(nodes)} structural nodes -> {len(semantic_nodes)} semantic nodes")

    with open("semantic_nodes.txt", mode="w+", encoding="utf-8") as f:
        for node in semantic_nodes:
            f.write(f"{node.get_metadata_str()}\n{node.get_content()}\n\n============================\n\n")