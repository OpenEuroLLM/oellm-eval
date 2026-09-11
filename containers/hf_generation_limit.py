"""Give the harness request's token budget precedence over model defaults."""
import ast


def patch(text):
    anchor='        # build stopping criteria\n'
    if text.count(anchor)!=1 or 'Some checkpoint exports omit EOS' not in text:
        raise ValueError('Unexpected HF adapter; reconstruct the pinned sources first')
    addition='''        # The harness passes the task budget as max_length. A saved model
        # max_new_tokens otherwise silently takes precedence in Transformers.
        # Keep explicit per-request overrides; never mutate generation_config.
        generation_kwargs.setdefault("max_new_tokens", None)

'''
    if addition in text:raise ValueError('Generation-limit correction already installed')
    updated=text.replace(anchor,addition+anchor)
    ast.parse(updated)
    return updated
