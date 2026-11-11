# MP2 Reflection

## What worked
Implementing a strict Input Sanitation & Guardrail Filter before the vector search step proved to be extremely effective. Adding a custom `is_valid_query()` filter—checking for alphanumeric density, minimum character length, and alphabet structure—allowed the pipeline to reject syntactic noise (like `.`, `&`, or whitespace) instantly without wasting API tokens or producing misleading output. Additionally, our multi-hop query tests successfully synthesized information across distinct story files (linking John Clay's pawnshop tunnel in *The Red-Headed League* with Dr. Roylott's motives in *The Speckled Band*) with proper source citations.

## What didn't work
During edge-case testing, passing punctuation inputs like `.` or `&` caused the pipeline to execute vector search on semantic noise, returning arbitrary chunks that led `gpt-4o-mini` to generate unrequested answers. Additionally, because Qdrant was configured in embedded local storage mode (`path="./qdrant_data"`), Python's garbage collector attempted to deallocate database resources during interpreter teardown, triggering an unhandled `sys.meta_path is None` deallocator traceback upon script exit.

## What I'd change
To fix the shutdown traceback, I implemented explicit resource lifecycle management by wrapping the execution dispatcher in a `try...finally` block that calls `client_qdrant.close()`. With another 5 hours, I would extend the input guardrails to include an LLM-based Query Expansion step that automatically rewrites short, bare entity keywords (e.g., "John Clay") into full, descriptive questions prior to vector retrieval to improve semantic context matching.

## One surprise
I was genuinely surprised by how effectively `text-embedding-3-small` and `gpt-4o-mini` synthesized information across multiple distinct source documents on hard multi-hop queries. When asking complex questions that required linking facts across *The Red-Headed League* and *The Speckled Band*, the vector index retrieved the exact relevant context from both separate files, and the LLM correctly generated a clear, comparative response with accurate multi-source citations without cross-contaminating facts.