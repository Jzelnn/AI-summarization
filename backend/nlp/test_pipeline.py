from nlp.pipeline import full_pipeline

text = """
Lionel Messi scored a stunning goal last night and led his team to victory.
"""

result = full_pipeline(text, "short")

print(result)
