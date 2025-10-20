PROMPT = """[ROLE] You are a Vietnamese hallucination classifier.
[GOAL] Determine whether the ANSWER contains hallucination relative to the CONTEXT and assign exactly one of 3 labels.

[CONTEXT]
{}

[QUESTION]
{}

[ANSWER]
{}

[GUIDELINES]
- Rely only on the [CONTEXT]; do not use outside knowledge or speculation.
- Label definitions:
  • class 1: no → Consistent with the context; does not introduce outside information.
  • class 2: intrinsic → Contradicts, distorts, or misinterprets information present in the context.
  • class 3: extrinsic → Adds information not present in the context (even if factually true elsewhere).
- Decision rules:
  • If partly correct but introduces content not in the context → extrinsic.
  • If it asserts something that conflicts with or corrupts the context → intrinsic.
  • If omissions are minor without being wrong or adding outside info → no.
- Prefer evidence by quoting or referencing the context; do not judge truth beyond the context.

Classification result:
The correct answer is: class {}"""

