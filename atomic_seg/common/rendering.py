"""Escaping and media cards for static dataset review pages."""

import html


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def image_card(href: str, caption: str) -> str:
    return f'<figure><img src="{escape(href)}" loading="lazy"><figcaption>{escape(caption)}</figcaption></figure>'
