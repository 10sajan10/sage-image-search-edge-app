# Architecture — Image Search at the Edge

Image Search at the Edge is an offline-first, multimodal image-search system that runs
entirely on a single **NVIDIA Jetson AGX Thor**. Every image is stored as several
independent representations — an image vector, a caption vector, and a lexical index — and
every query is scored against each and fused into a single ranked list. All model and
database work happens on the device, with no runtime internet dependency.

## Ingestion — building the index

The raw image feeds two branches in parallel: a CLIP image encoder (the image vector) and a
local vision-language captioner. The generated caption then feeds its own two branches: a
CLIP text encoder (the caption vector) and a BM25 lexical index. All three representations
land in the vector database, keyed to the same image.

<p align="center"><svg xmlns="http://www.w3.org/2000/svg" width="680" viewBox="0 0 680 405" role="img">
<title>Weaviate database ingestion</title>
<style>
text{font-family:'Anthropic Sans',-apple-system,'Segoe UI',Roboto,sans-serif;}
.th{font-size:14px;font-weight:500;}
.ts{font-size:12px;}
.arr{stroke:#5F5E5A;stroke-width:1.5;fill:none;stroke-linecap:round;}
.lbl{fill:#5F5E5A;}
rect.box{stroke-width:0.5;}
.gray rect{fill:#F1EFE8;stroke:#5F5E5A;} .gray text{fill:#2C2C2A;}
.purple rect{fill:#EEEDFE;stroke:#534AB7;} .purple text{fill:#26215C;}
.teal rect{fill:#E1F5EE;stroke:#0F6E56;} .teal text{fill:#04342C;}
.amber rect{fill:#FAEEDA;stroke:#854F0B;} .amber text{fill:#412402;}
.blue rect{fill:#E6F1FB;stroke:#185FA5;} .blue text{fill:#042C53;}
@media (prefers-color-scheme:dark){
 .arr{stroke:#9c9a92;} .lbl{fill:#9c9a92;}
 .gray rect{fill:#444441;stroke:#B4B2A9;} .gray text{fill:#D3D1C7;}
 .purple rect{fill:#3C3489;stroke:#AFA9EC;} .purple text{fill:#CECBF6;}
 .teal rect{fill:#085041;stroke:#5DCAA5;} .teal text{fill:#9FE1CB;}
 .amber rect{fill:#633806;stroke:#EF9F27;} .amber text{fill:#FAC775;}
 .blue rect{fill:#0C447C;stroke:#85B7EB;} .blue text{fill:#B5D4F4;}
}
</style>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
<g class="gray">
<rect class="box" x="260" y="28" width="160" height="54" rx="8"/>
<text class="th" x="340.0" y="46.0" text-anchor="middle" dominant-baseline="central">Camera capture</text>
<text class="ts" x="340.0" y="64.0" text-anchor="middle" dominant-baseline="central">Jetson AGX Thor</text>
</g>
<g class="purple">
<rect class="box" x="252" y="116" width="176" height="54" rx="8"/>
<text class="th" x="340.0" y="134.0" text-anchor="middle" dominant-baseline="central">Gemma captioner</text>
<text class="ts" x="340.0" y="152.0" text-anchor="middle" dominant-baseline="central">gemma4:e2b, ~50 words</text>
</g>
<g class="teal">
<rect class="box" x="40" y="210" width="186" height="54" rx="8"/>
<text class="th" x="133.0" y="228.0" text-anchor="middle" dominant-baseline="central">Image encoder</text>
<text class="ts" x="133.0" y="246.0" text-anchor="middle" dominant-baseline="central">DFN5B-CLIP → vector</text>
</g>
<g class="teal">
<rect class="box" x="247" y="210" width="186" height="54" rx="8"/>
<text class="th" x="340.0" y="228.0" text-anchor="middle" dominant-baseline="central">Caption encoder</text>
<text class="ts" x="340.0" y="246.0" text-anchor="middle" dominant-baseline="central">DFN5B-CLIP, 77 tok</text>
</g>
<g class="amber">
<rect class="box" x="454" y="210" width="186" height="54" rx="8"/>
<text class="th" x="547.0" y="228.0" text-anchor="middle" dominant-baseline="central">BM25 index</text>
<text class="ts" x="547.0" y="246.0" text-anchor="middle" dominant-baseline="central">lexical, caption^2</text>
</g>
<g class="blue">
<rect class="box" x="185" y="305" width="310" height="56" rx="8"/>
<text class="th" x="340.0" y="324.0" text-anchor="middle" dominant-baseline="central">Weaviate database</text>
<text class="ts" x="340.0" y="342.0" text-anchor="middle" dominant-baseline="central">3 representations per image</text>
</g>
<line x1="340" y1="82" x2="340" y2="112" class="arr" marker-end="url(#arrow)"/>
<path d="M260 55 L133 55 L133 206" class="arr" marker-end="url(#arrow)"/>
<text class="ts lbl" x="188" y="47" text-anchor="middle">image</text>
<line x1="340" y1="170" x2="340" y2="206" class="arr" marker-end="url(#arrow)"/>
<text class="ts lbl" x="356" y="190" text-anchor="start">caption</text>
<path d="M428 143 L547 143 L547 206" class="arr" marker-end="url(#arrow)"/>
<text class="ts lbl" x="500" y="134" text-anchor="middle">caption</text>
<path d="M133 264 L133 333 L181 333" class="arr" marker-end="url(#arrow)"/>
<line x1="340" y1="264" x2="340" y2="301" class="arr" marker-end="url(#arrow)"/>
<path d="M547 264 L547 333 L499 333" class="arr" marker-end="url(#arrow)"/>
</svg></p>

## Search — querying and fusion

The query splits the same way the image did. The encoded query vector drives cosine
similarity against both the image vector and the caption vector, while the raw query terms
drive the BM25 lexical leg. Each leg is normalized independently, then combined with fixed
fusion weights. The top results come straight out of the fused score.

<p align="center"><svg xmlns="http://www.w3.org/2000/svg" width="680" viewBox="0 0 680 495" role="img">
<title>Search and fusion</title>
<style>
text{font-family:'Anthropic Sans',-apple-system,'Segoe UI',Roboto,sans-serif;}
.th{font-size:14px;font-weight:500;}
.ts{font-size:12px;}
.arr{stroke:#5F5E5A;stroke-width:1.5;fill:none;stroke-linecap:round;}
.lbl{fill:#5F5E5A;}
rect.box{stroke-width:0.5;}
.gray rect{fill:#F1EFE8;stroke:#5F5E5A;} .gray text{fill:#2C2C2A;}
.purple rect{fill:#EEEDFE;stroke:#534AB7;} .purple text{fill:#26215C;}
.teal rect{fill:#E1F5EE;stroke:#0F6E56;} .teal text{fill:#04342C;}
.amber rect{fill:#FAEEDA;stroke:#854F0B;} .amber text{fill:#412402;}
.blue rect{fill:#E6F1FB;stroke:#185FA5;} .blue text{fill:#042C53;}
@media (prefers-color-scheme:dark){
 .arr{stroke:#9c9a92;} .lbl{fill:#9c9a92;}
 .gray rect{fill:#444441;stroke:#B4B2A9;} .gray text{fill:#D3D1C7;}
 .purple rect{fill:#3C3489;stroke:#AFA9EC;} .purple text{fill:#CECBF6;}
 .teal rect{fill:#085041;stroke:#5DCAA5;} .teal text{fill:#9FE1CB;}
 .amber rect{fill:#633806;stroke:#EF9F27;} .amber text{fill:#FAC775;}
 .blue rect{fill:#0C447C;stroke:#85B7EB;} .blue text{fill:#B5D4F4;}
}
</style>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
<g class="gray">
<rect class="box" x="260" y="28" width="160" height="54" rx="8"/>
<text class="th" x="340.0" y="46.0" text-anchor="middle" dominant-baseline="central">Search query</text>
<text class="ts" x="340.0" y="64.0" text-anchor="middle" dominant-baseline="central">natural-language text</text>
</g>
<g class="purple">
<rect class="box" x="250" y="116" width="180" height="54" rx="8"/>
<text class="th" x="340.0" y="134.0" text-anchor="middle" dominant-baseline="central">Query encoder</text>
<text class="ts" x="340.0" y="152.0" text-anchor="middle" dominant-baseline="central">CLIP text → vector</text>
</g>
<g class="teal">
<rect class="box" x="40" y="210" width="186" height="54" rx="8"/>
<text class="th" x="133.0" y="228.0" text-anchor="middle" dominant-baseline="central">Image vector sim</text>
<text class="ts" x="133.0" y="246.0" text-anchor="middle" dominant-baseline="central">cosine vs image_clip</text>
</g>
<g class="teal">
<rect class="box" x="247" y="210" width="186" height="54" rx="8"/>
<text class="th" x="340.0" y="228.0" text-anchor="middle" dominant-baseline="central">Caption vector sim</text>
<text class="ts" x="340.0" y="246.0" text-anchor="middle" dominant-baseline="central">cosine vs caption_clip</text>
</g>
<g class="amber">
<rect class="box" x="454" y="210" width="186" height="54" rx="8"/>
<text class="th" x="547.0" y="228.0" text-anchor="middle" dominant-baseline="central">BM25 match</text>
<text class="ts" x="547.0" y="246.0" text-anchor="middle" dominant-baseline="central">terms vs caption</text>
</g>
<g class="blue">
<rect class="box" x="235" y="305" width="210" height="56" rx="8"/>
<text class="th" x="340.0" y="324.0" text-anchor="middle" dominant-baseline="central">Weighted fusion</text>
<text class="ts" x="340.0" y="342.0" text-anchor="middle" dominant-baseline="central">0.60 / 0.25 / 0.15</text>
</g>
<g class="gray">
<rect class="box" x="255" y="400" width="170" height="54" rx="8"/>
<text class="th" x="340.0" y="418.0" text-anchor="middle" dominant-baseline="central">Top-k results</text>
<text class="ts" x="340.0" y="436.0" text-anchor="middle" dominant-baseline="central">ranked, K = 25</text>
</g>
<line x1="340" y1="82" x2="340" y2="112" class="arr" marker-end="url(#arrow)"/>
<path d="M340 170 L340 206" class="arr" marker-end="url(#arrow)"/>
<path d="M340 188 L133 188 L133 206" class="arr" marker-end="url(#arrow)"/>
<text class="ts lbl" x="205" y="180" text-anchor="middle">query vector</text>
<path d="M420 55 L547 55 L547 206" class="arr" marker-end="url(#arrow)"/>
<text class="ts lbl" x="500" y="47" text-anchor="middle">query terms</text>
<path d="M133 264 L133 333 L231 333" class="arr" marker-end="url(#arrow)"/>
<line x1="340" y1="264" x2="340" y2="301" class="arr" marker-end="url(#arrow)"/>
<path d="M547 264 L547 333 L449 333" class="arr" marker-end="url(#arrow)"/>
<line x1="340" y1="361" x2="340" y2="396" class="arr" marker-end="url(#arrow)"/>
</svg></p>

## A newer configuration: long captions

A later configuration keeps the same three-leg shape but lifts a key constraint. The
earlier CLIP text encoder capped captions at 77 tokens, which forced a hard tradeoff:
detailed captions were too long to embed (reachable only through keyword matching), so
captions had to be shrunk to fit — embeddable, but thin. Switching to a long-context
embedder (**jina-clip-v2**, 8192 tokens) removes that ceiling, so the caption can grow to
~250 words *and* be embedded. Captioner "thinking" is turned off since it consumed the
token budget for no measurable gain, and storage moves to **Qdrant**.

<p align="center"><svg xmlns="http://www.w3.org/2000/svg" width="680" viewBox="0 0 680 405" role="img">
<title>Qdrant database ingestion</title>
<style>
text{font-family:'Anthropic Sans',-apple-system,'Segoe UI',Roboto,sans-serif;}
.th{font-size:14px;font-weight:500;}
.ts{font-size:12px;}
.arr{stroke:#5F5E5A;stroke-width:1.5;fill:none;stroke-linecap:round;}
.lbl{fill:#5F5E5A;}
rect.box{stroke-width:0.5;}
.gray rect{fill:#F1EFE8;stroke:#5F5E5A;} .gray text{fill:#2C2C2A;}
.purple rect{fill:#EEEDFE;stroke:#534AB7;} .purple text{fill:#26215C;}
.teal rect{fill:#E1F5EE;stroke:#0F6E56;} .teal text{fill:#04342C;}
.amber rect{fill:#FAEEDA;stroke:#854F0B;} .amber text{fill:#412402;}
.blue rect{fill:#E6F1FB;stroke:#185FA5;} .blue text{fill:#042C53;}
@media (prefers-color-scheme:dark){
 .arr{stroke:#9c9a92;} .lbl{fill:#9c9a92;}
 .gray rect{fill:#444441;stroke:#B4B2A9;} .gray text{fill:#D3D1C7;}
 .purple rect{fill:#3C3489;stroke:#AFA9EC;} .purple text{fill:#CECBF6;}
 .teal rect{fill:#085041;stroke:#5DCAA5;} .teal text{fill:#9FE1CB;}
 .amber rect{fill:#633806;stroke:#EF9F27;} .amber text{fill:#FAC775;}
 .blue rect{fill:#0C447C;stroke:#85B7EB;} .blue text{fill:#B5D4F4;}
}
</style>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
<g class="gray">
<rect class="box" x="260" y="28" width="160" height="54" rx="8"/>
<text class="th" x="340.0" y="46.0" text-anchor="middle" dominant-baseline="central">Camera capture</text>
<text class="ts" x="340.0" y="64.0" text-anchor="middle" dominant-baseline="central">Jetson AGX Thor</text>
</g>
<g class="purple">
<rect class="box" x="252" y="116" width="176" height="54" rx="8"/>
<text class="th" x="340.0" y="134.0" text-anchor="middle" dominant-baseline="central">Gemma captioner</text>
<text class="ts" x="340.0" y="152.0" text-anchor="middle" dominant-baseline="central">gemma4:e2b, ~250 words</text>
</g>
<g class="teal">
<rect class="box" x="40" y="210" width="186" height="54" rx="8"/>
<text class="th" x="133.0" y="228.0" text-anchor="middle" dominant-baseline="central">Image encoder</text>
<text class="ts" x="133.0" y="246.0" text-anchor="middle" dominant-baseline="central">jina-clip-v2 → vector</text>
</g>
<g class="teal">
<rect class="box" x="247" y="210" width="186" height="54" rx="8"/>
<text class="th" x="340.0" y="228.0" text-anchor="middle" dominant-baseline="central">Caption encoder</text>
<text class="ts" x="340.0" y="246.0" text-anchor="middle" dominant-baseline="central">jina-clip-v2, 8192 tok</text>
</g>
<g class="amber">
<rect class="box" x="454" y="210" width="186" height="54" rx="8"/>
<text class="th" x="547.0" y="228.0" text-anchor="middle" dominant-baseline="central">BM25 index</text>
<text class="ts" x="547.0" y="246.0" text-anchor="middle" dominant-baseline="central">lexical, caption</text>
</g>
<g class="blue">
<rect class="box" x="185" y="305" width="310" height="56" rx="8"/>
<text class="th" x="340.0" y="324.0" text-anchor="middle" dominant-baseline="central">Qdrant database</text>
<text class="ts" x="340.0" y="342.0" text-anchor="middle" dominant-baseline="central">UUID5 point IDs</text>
</g>
<line x1="340" y1="82" x2="340" y2="112" class="arr" marker-end="url(#arrow)"/>
<path d="M260 55 L133 55 L133 206" class="arr" marker-end="url(#arrow)"/>
<text class="ts lbl" x="188" y="47" text-anchor="middle">image</text>
<line x1="340" y1="170" x2="340" y2="206" class="arr" marker-end="url(#arrow)"/>
<text class="ts lbl" x="356" y="190" text-anchor="start">caption</text>
<path d="M428 143 L547 143 L547 206" class="arr" marker-end="url(#arrow)"/>
<text class="ts lbl" x="500" y="134" text-anchor="middle">caption</text>
<path d="M133 264 L133 333 L181 333" class="arr" marker-end="url(#arrow)"/>
<line x1="340" y1="264" x2="340" y2="301" class="arr" marker-end="url(#arrow)"/>
<path d="M547 264 L547 333 L499 333" class="arr" marker-end="url(#arrow)"/>
</svg></p>

## Configurations at a glance

| | Baseline | Dual-embedding | Long-caption |
|---|---|---|---|
| Caption model | `gemma-3-4b-it` | `gemma4:e2b`, thinking on | `gemma4:e2b`, thinking off |
| Caption length | ~150 words | ~50 words | ~250 words |
| Embedder | DFN5B-CLIP (77 tok) | DFN5B-CLIP (77 tok) | jina-clip-v2 (8192 tok) |
| Vectors stored | image only | image + caption | image + caption |
| Fusion | 75% image + 25% BM25 | 60% image + 25% caption + 15% BM25 | image + caption + BM25 |
| Database | Weaviate | Weaviate | Qdrant |

Color key for the diagrams: gray = source/output, purple = neural model, teal = CLIP vector
operations, amber = BM25 lexical, blue = storage/fusion.

## How it performs

Across five public image-search benchmarks, the edge system is compared against a set of
datacenter reference systems on a single composite score. Edge configurations are shown in
teal.

<p align="center"><svg xmlns="http://www.w3.org/2000/svg" width="680" viewBox="0 0 680 350" role="img"><title>Overall primary leaderboard (MRR + Success@25)</title><style>
text{font-family:'Anthropic Sans',-apple-system,'Segoe UI',Roboto,sans-serif;}
.ttl{font-size:15px;font-weight:500;} .val{font-size:12px;font-weight:500;}
.axl{font-size:12px;} .grid{stroke:#D3D1C7;stroke-width:0.5;} .axis{stroke:#888780;stroke-width:1;}
.ttl,.val,.axl{fill:#2C2C2A;}
.ref rect{fill:#B5D4F4;stroke:#185FA5;stroke-width:0.5;}
.edge rect{fill:#5DCAA5;stroke:#0F6E56;stroke-width:0.5;}
@media (prefers-color-scheme:dark){
 .ttl,.val,.axl{fill:#D3D1C7;} .grid{stroke:#444441;} .axis{stroke:#888780;}
 .ref rect{fill:#185FA5;stroke:#85B7EB;} .edge rect{fill:#0F6E56;stroke:#5DCAA5;}
}
</style><text class="ttl" x="360" y="28" text-anchor="middle">Overall primary leaderboard (MRR + Success@25)</text><line class="grid" x1="70" y1="300.0" x2="650" y2="300.0"/><text class="axl" x="60" y="304.0" text-anchor="end">0.0</text><line class="grid" x1="70" y1="216.7" x2="650" y2="216.7"/><text class="axl" x="60" y="220.7" text-anchor="end">0.2</text><line class="grid" x1="70" y1="133.3" x2="650" y2="133.3"/><text class="axl" x="60" y="137.3" text-anchor="end">0.4</text><line class="grid" x1="70" y1="50.0" x2="650" y2="50.0"/><text class="axl" x="60" y="54.0" text-anchor="end">0.6</text><line class="axis" x1="70" y1="300" x2="650" y2="300"/><g class="ref"><rect x="88.4" y="82.1" width="59.9" height="217.9" rx="3"/></g><text class="val" x="118.3" y="76.1" text-anchor="middle">0.523</text><text class="axl" x="118.3" y="318.0" text-anchor="middle">v10</text><g class="ref"><rect x="185.0" y="83.3" width="59.9" height="216.7" rx="3"/></g><text class="val" x="215.0" y="77.3" text-anchor="middle">0.520</text><text class="axl" x="215.0" y="318.0" text-anchor="middle">v11</text><g class="edge"><rect x="281.7" y="84.6" width="59.9" height="215.4" rx="3"/></g><text class="val" x="311.7" y="78.6" text-anchor="middle">0.517</text><text class="axl" x="311.7" y="318.0" text-anchor="middle">edge_v1</text><g class="ref"><rect x="378.4" y="96.7" width="59.9" height="203.3" rx="3"/></g><text class="val" x="408.3" y="90.7" text-anchor="middle">0.488</text><text class="axl" x="408.3" y="318.0" text-anchor="middle">v12</text><g class="ref"><rect x="475.0" y="111.2" width="59.9" height="188.8" rx="3"/></g><text class="val" x="505.0" y="105.2" text-anchor="middle">0.453</text><text class="axl" x="505.0" y="318.0" text-anchor="middle">baseline</text><g class="edge"><rect x="571.7" y="136.7" width="59.9" height="163.3" rx="3"/></g><text class="val" x="601.7" y="130.7" text-anchor="middle">0.392</text><text class="axl" x="601.7" y="318.0" text-anchor="middle">edge_v2</text></svg></p>

<p align="center"><svg xmlns="http://www.w3.org/2000/svg" width="680" viewBox="0 0 680 350" role="img"><title>Overall primary + diversity leaderboard</title><style>
text{font-family:'Anthropic Sans',-apple-system,'Segoe UI',Roboto,sans-serif;}
.ttl{font-size:15px;font-weight:500;} .val{font-size:12px;font-weight:500;}
.axl{font-size:12px;} .grid{stroke:#D3D1C7;stroke-width:0.5;} .axis{stroke:#888780;stroke-width:1;}
.ttl,.val,.axl{fill:#2C2C2A;}
.ref rect{fill:#B5D4F4;stroke:#185FA5;stroke-width:0.5;}
.edge rect{fill:#5DCAA5;stroke:#0F6E56;stroke-width:0.5;}
@media (prefers-color-scheme:dark){
 .ttl,.val,.axl{fill:#D3D1C7;} .grid{stroke:#444441;} .axis{stroke:#888780;}
 .ref rect{fill:#185FA5;stroke:#85B7EB;} .edge rect{fill:#0F6E56;stroke:#5DCAA5;}
}
</style><text class="ttl" x="360" y="28" text-anchor="middle">Overall primary + diversity leaderboard</text><line class="grid" x1="70" y1="300.0" x2="650" y2="300.0"/><text class="axl" x="60" y="304.0" text-anchor="end">0.0</text><line class="grid" x1="70" y1="216.7" x2="650" y2="216.7"/><text class="axl" x="60" y="220.7" text-anchor="end">0.2</text><line class="grid" x1="70" y1="133.3" x2="650" y2="133.3"/><text class="axl" x="60" y="137.3" text-anchor="end">0.4</text><line class="grid" x1="70" y1="50.0" x2="650" y2="50.0"/><text class="axl" x="60" y="54.0" text-anchor="end">0.6</text><line class="axis" x1="70" y1="300" x2="650" y2="300"/><g class="ref"><rect x="88.4" y="120.8" width="59.9" height="179.2" rx="3"/></g><text class="val" x="118.3" y="114.8" text-anchor="middle">0.430</text><text class="axl" x="118.3" y="318.0" text-anchor="middle">v10</text><g class="ref"><rect x="185.0" y="120.8" width="59.9" height="179.2" rx="3"/></g><text class="val" x="215.0" y="114.8" text-anchor="middle">0.430</text><text class="axl" x="215.0" y="318.0" text-anchor="middle">v11</text><g class="edge"><rect x="281.7" y="128.8" width="59.9" height="171.2" rx="3"/></g><text class="val" x="311.7" y="122.8" text-anchor="middle">0.411</text><text class="axl" x="311.7" y="318.0" text-anchor="middle">edge_v1</text><g class="ref"><rect x="378.4" y="130.0" width="59.9" height="170.0" rx="3"/></g><text class="val" x="408.3" y="124.0" text-anchor="middle">0.408</text><text class="axl" x="408.3" y="318.0" text-anchor="middle">v12</text><g class="ref"><rect x="475.0" y="151.7" width="59.9" height="148.3" rx="3"/></g><text class="val" x="505.0" y="145.7" text-anchor="middle">0.356</text><text class="axl" x="505.0" y="318.0" text-anchor="middle">baseline</text><g class="edge"><rect x="571.7" y="157.5" width="59.9" height="142.5" rx="3"/></g><text class="val" x="601.7" y="151.5" text-anchor="middle">0.342</text><text class="axl" x="601.7" y="318.0" text-anchor="middle">edge_v2</text></svg></p>

Despite running fully on-device, the edge baseline ranks **3rd of 6** on the primary
composite (0.517, within ~0.006 of the top reference systems) and posts the **highest
Success@25 of any system (0.697)** — it most reliably surfaces a relevant image in the top
results. Its lower MRR, from having no reranker, is what keeps it just behind the two
leaders.
