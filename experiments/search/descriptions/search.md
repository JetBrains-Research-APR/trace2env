# AgentWorldBench search

A web research session: a search agent answers one question per trajectory (many in Chinese, about transit lines,
companies, people, places, products, regulations) by calling two tools, and the world model plays the search engine and
the web. There is no task instruction in the prompts; the question is implicit in the queries. The observation after
every call is the tool's textual result, and the session's state is what has been searched, which pages have been
fetched and what they said, so that later results stay consistent with earlier ones.

Actions are JSON tool calls `{"name": <tool>, "arguments": {...}}`:

- `web_search` with `{"queries": ["q1", "q2", ...]}` (a batch; occasionally `{"query": "..."}`). The observation is a
  ranked result list, ten results per query in sequence, each formatted as `[n] <url>` on one line, then `Title: ...`,
  `Snippet: ...`, `Date: <Mon d, yyyy or N/A>`, separated by blank lines. Numbering runs `[1]`, `[2]`, ... across the
  batch; URLs are real-looking (Wikipedia, official sites, news outlets, forums, government and academic pages) and
  match the query's language and topic; snippets tease the relevant facts without fully answering; reference pages are
  undated (`N/A`), news and blog results carry dates.
- `web_extractor` with `{"url": "<url>"}`, normally a URL returned by an earlier search. The observation is the page's
  text in Markdown: headings with `#`, tables with `|`, lists with `-` or `1.`, quotes with `>`, boilerplate removed;
  Wikipedia pages have infobox facts and section headers, news pages a headline, author, date and body, official pages
  structured specifications. Content must agree with the snippet that advertised the page. An inaccessible page yields
  an error message instead.
- `dict_memory` (rare) with `add` / `update` / `retrieve` / `retrieve_all` / `remove` / `remove_all` operations over a
  persistent key-value store that answers `Successfully added items of [keys] to memory.`-style confirmations, the
  stored values, or `Key not found`.

Behaviour: results and page contents are factual real-world information (no invented entities, numbers or citations);
repeated or refined queries return overlapping results; an extracted page reflects the earlier snippet for the same
URL; empty results and malformed requests produce short error texts. Each transition is teacher-forced: the next
observation is the real tool response recorded in the session.
