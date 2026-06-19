# Frozen English corpus v1

This corpus contains 60 source items: 30 mathematics items and 30 factual
knowledge items. Each item renders into neutral, correct-belief, and
wrong-belief conditions using `english-counterfactual-template-v1`.

- Rendered dataset SHA-256: `b5b2366192f9e168bb5c74db884eba03cbab2eab17bd9d87d9c3ee45a72d970c`
- Frozen split SHA-256: `a72800224e0ecb6855ca16b44b0151a94909d2eb7c52a79424cc9b59ff0d8fcc`
- Train: 40 item-disjoint source items
- Test: 20 item-disjoint source items
- Correct-answer positions: 30 A and 30 B

Only the `User belief:` line changes across an item's three rendered prompts.
The factual label and both answer options remain fixed.
