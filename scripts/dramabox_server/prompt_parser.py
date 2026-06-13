"""Prompt parser for the DramaBox inference server.

Splits multi-dialogue prompts at double-quote block boundaries so each
dialogue segment can be dispatched to a separate GPU for parallel generation.
Speaker prefix (persona / style description) is preserved on every segment.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Import DramaBox's duration estimator
_DRAMABOX_SRC = str(Path(__file__).resolve().parent.parent.parent / "DramaBox" / "src")
if _DRAMABOX_SRC not in sys.path:
    sys.path.insert(0, _DRAMABOX_SRC)

from duration_estimator import estimate_speech_duration


@dataclass
class Segment:
    index: int
    text: str
    estimated_duration: float


# Matches the leading speaker description up to the first comma before an
# opening double-quote.  Same regex as DramaBox's text_chunker._PREFIX_RE.
_PREFIX_RE = re.compile(r'^([^"\']{3,}?)(,\s*)(?=["\'])', re.DOTALL)


def extract_speaker_prefix(prompt: str) -> tuple[Optional[str], str]:
    """Return (prefix, body).  prefix is the persona/style description."""
    m = _PREFIX_RE.match(prompt)
    if not m:
        return None, prompt
    return m.group(1).strip(), prompt[m.end():]


def split_at_quote_blocks(body: str) -> List[str]:
    """Split body text so each segment contains exactly one double-quoted
    dialogue block plus trailing narration/stage directions.

    Each segment runs from one opening quote through the closing quote and
    any narration that follows, up to (but not including) the next opening
    quote.  Leading narration before the very first quote is attached to
    segment 0.

    Example:
        '"Hello." She pauses. "How are you?" She laughs. "Great!"'
    becomes:
        ['"Hello." She pauses.', '"How are you?" She laughs.', '"Great!"']
    """
    # Find opening-quote positions
    quote_starts = []
    in_quote = False
    for i, ch in enumerate(body):
        if ch == '"':
            if not in_quote:
                quote_starts.append(i)
            in_quote = not in_quote

    if len(quote_starts) <= 1:
        return [body.strip()] if body.strip() else []

    # Each segment: from this opening quote to just before the next opening quote.
    # First segment also includes any leading narration before the first quote.
    segments: List[str] = []
    for idx, qstart in enumerate(quote_starts):
        seg_start = 0 if idx == 0 else qstart
        seg_end = quote_starts[idx + 1] if idx + 1 < len(quote_starts) else len(body)
        seg_text = body[seg_start:seg_end].strip()
        if seg_text:
            segments.append(seg_text)

    return segments


def _reassemble(prefix: Optional[str], segment_body: str) -> str:
    """Re-attach the speaker prefix to a segment body."""
    if not prefix:
        return segment_body
    if segment_body.lstrip().startswith(('"', "'")):
        return f'{prefix}, {segment_body}'
    return f'{prefix}. {segment_body}'


def parse_prompt(prompt: str, duration_multiplier: float = 1.1) -> List[Segment]:
    """Parse a DramaBox prompt into segments for parallel generation.

    Each segment contains one double-quoted dialogue block (plus any
    surrounding narration/stage directions). The speaker prefix is
    re-attached to every segment.

    Returns a list of Segment objects with estimated durations.
    """
    prefix, body = extract_speaker_prefix(prompt)
    parts = split_at_quote_blocks(body)

    if len(parts) <= 1:
        # Single segment or no quotes found — return the whole prompt
        dur = estimate_speech_duration(prompt) * duration_multiplier
        return [Segment(index=0, text=prompt, estimated_duration=dur)]

    segments = []
    for i, part in enumerate(parts):
        text = _reassemble(prefix, part)
        dur = estimate_speech_duration(text) * duration_multiplier
        segments.append(Segment(index=i, text=text, estimated_duration=dur))

    return segments
