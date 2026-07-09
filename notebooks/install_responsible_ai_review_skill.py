# Databricks notebook source
# MAGIC %md
# MAGIC # Install the Responsible AI Review skill for Genie Code
# MAGIC
# MAGIC This notebook copies the **responsible-ai-review** skill into your personal
# MAGIC Genie Code skills folder (`/Workspace/Users/<you>/.assistant/skills/`), so Genie
# MAGIC Code can use it to review a project against the ten UK Government AI Principles
# MAGIC and produce a RAG scorecard.
# MAGIC
# MAGIC **How to use:** attach to any cluster/serverless and *Run all*. Then open Genie
# MAGIC Code and ask, e.g. *"Review this project against the responsible AI principles
# MAGIC and give me a score."*
# MAGIC
# MAGIC Source of truth: `/Workspace/Shared/responsible-ai-review-skill/`
# MAGIC (mirrored from `.claude/skills/responsible-ai-review/` in the responsible-ai repo).

# COMMAND ----------

import posixpath
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat, ExportFormat

SOURCE = "/Workspace/Shared/responsible-ai-review-skill"
SKILL_NAME = "responsible-ai-review"

w = WorkspaceClient()
me = w.current_user.me().user_name
dest_root = f"/Workspace/Users/{me}/.assistant/skills/{SKILL_NAME}"
print(f"Installing '{SKILL_NAME}' for {me}\n  from: {SOURCE}\n  to:   {dest_root}")

# COMMAND ----------

def iter_files(root):
    """Yield every FILE path under a workspace directory, recursively."""
    for obj in w.workspace.list(root):
        if str(obj.object_type) == "ObjectType.DIRECTORY" or obj.object_type.value == "DIRECTORY":
            yield from iter_files(obj.path)
        else:
            yield obj.path

count = 0
for src in iter_files(SOURCE):
    rel = posixpath.relpath(src, SOURCE)
    dest = posixpath.join(dest_root, rel)
    content = w.workspace.export(path=src, format=ExportFormat.AUTO).content  # base64
    w.workspace.mkdirs(posixpath.dirname(dest))
    w.workspace.import_(path=dest, content=content, format=ImportFormat.AUTO, overwrite=True)
    print(f"  installed {rel}")
    count += 1

print(f"\nDone — {count} file(s) installed to {dest_root}")
print("Open Genie Code and try: 'Review this project against the responsible AI principles and give me a score.'")
