"""The newsroom: researched, stats-grounded articles for the top of each card.

    wire     free news (Google News + outlet RSS) for the two fighters
    claims   DeepSeek pulls sourced storylines and verbatim quotes out of each story
    facts    the model's numbers for the bout, the only source of stats
    writer   an editor plans the article, a writer drafts it
    checker  every number, quote and the pick are verified before it publishes
    pipeline runs the above per fight; `python -m app.services.ufc.newsroom`

Fights below the top NEWSROOM_TOP_N keep the single-call preview in preview_service.
"""
