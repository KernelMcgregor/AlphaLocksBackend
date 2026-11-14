"""The news desk: trending UFC stories, found and written twice a day.

    sources   free headline sweeps (Google News + outlet RSS), queries built from the DB
    entities  which UFC fighters a headline is about
    topics    headlines grouped into stories, scored, matched to stories already running
    labeler   one DeepSeek call: title, kind, newsworthy, a follow-up search
    desk      runs it; `python -m app.services.ufc.newsroom.desk`

Research, writing and checking reuse the preview newsroom (wire, claims, writer, checker).
"""
