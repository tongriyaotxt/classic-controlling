# -*- coding: utf-8 -*-
"""Minimal md -> html for the zhihu post (headers, bold, bullets, fenced code, hr, paragraphs)."""
import html, re, sys, pathlib

src = pathlib.Path(r"D:\项目\classic_controlling\docs\posts\zhihu_post.md").read_text(encoding="utf-8")
lines = src.splitlines()[1:]  # skip the H1 title line

def inline(t):
    t = html.escape(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    return t

out, i = [], 0
while i < len(lines):
    ln = lines[i]
    if ln.startswith("```"):
        i += 1
        buf = []
        while i < len(lines) and not lines[i].startswith("```"):
            buf.append(lines[i]); i += 1
        i += 1
        out.append("<pre>" + html.escape("\n".join(buf)) + "</pre>")
        continue
    if ln.startswith("## "):
        out.append(f"<h2>{inline(ln[3:])}</h2>")
    elif ln.strip() == "---":
        out.append("<hr>")
    elif ln.startswith("- "):
        items = []
        while i < len(lines) and lines[i].startswith("- "):
            items.append(f"<li>{inline(lines[i][2:])}</li>"); i += 1
        out.append("<ul>" + "".join(items) + "</ul>")
        continue
    elif ln.strip() == "":
        pass
    else:
        out.append(f"<p>{inline(ln)}</p>")
    i += 1

doc = ("<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>"
       + "\n".join(out) + "</body></html>")
outpath = pathlib.Path(r"D:\项目\classic_controlling\docs\posts\zhihu_post.html")
outpath.write_text(doc, encoding="utf-8")
print("written", outpath, len(doc))
