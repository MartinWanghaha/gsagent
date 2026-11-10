"""Resolve an organized publication directory to its compatible working view.

The working view uses directory links: source mesh/scene identities and existing
score request IDs remain stable while technical data stays out of the top level.
"""
import json
import os
from pathlib import Path
import re

MARKER='.mesh_layout.json'


def configuration(path):
    p=Path(path).absolute()
    file=p/MARKER
    return json.loads(file.read_text()) if file.exists() else None


def work_root(path):
    p=Path(path).absolute();cfg=configuration(p)
    return Path(cfg['work_root']) if cfg else p


def public_root(path):
    p=Path(path).absolute();cfg=configuration(p)
    return Path(cfg['public_root']) if cfg else p


def configure_sources(path):
    cfg=configuration(path)
    if cfg:
        from . import core
        core.RECOMPOSE=Path(cfg['public_root'])/'recomposition'


def figure_bucket(figures,kind):
    p=Path(figures).resolve()
    return p/kind if configuration(p.parent) else p


def document_links(text,source,destination):
    """Relocate local Markdown links without changing URLs, code, or scores."""
    source=Path(source);destination=Path(destination)
    def replace(match):
        target=match.group(2)
        if target.startswith(('#','http:','https:','mailto:')):return match.group(0)
        bare,sep,fragment=target.strip('<>').partition('#')
        if not bare:return match.group(0)
        absolute=Path(os.path.abspath(source.parent/bare)).resolve()
        relative=os.path.relpath(absolute,destination.parent)
        if sep:relative+='#'+fragment
        if ' ' in relative:relative='<'+relative+'>'
        return match.group(1)+relative+')'
    return re.sub(r'(!?\[[^\]]*\]\()([^\n)]+)\)',replace,text)


def publish_reports(path):
    cfg=configuration(path)
    if not cfg:return
    public=Path(cfg['public_root']);work=Path(cfg['work_root'])
    target=public/'reports';target.mkdir(exist_ok=True)
    for state in ['background','recomposition']:
        source=public/'results'/state/'report.md';dest=target/(state+'.md')
        if source.exists():dest.write_text(document_links(source.read_text(),source,dest))
