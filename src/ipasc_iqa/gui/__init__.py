"""Web GUI for ipasc_iqa.

Architecture (clean UI/function separation):

- ``service.py``  — the *function layer*: pure Python returning plain data
  structures. No FastAPI, no HTML. This is what you call/test in isolation.
- ``routes/``     — thin HTTP adapters: parse request -> call service ->
  render a template or return JSON. No business logic.
- ``templates/``  — the *UI layer*: Jinja2 HTML you hand-edit.
- ``static/``     — CSS / JS / images you hand-edit.

The whole GUI is optional; install with ``pip install ipasc_iqa[gui]``.
"""
