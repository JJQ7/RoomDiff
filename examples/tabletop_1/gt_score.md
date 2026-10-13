## GT score: tabletop_1 (outputs/tabletop_1)

| GT | object | type | result | prediction | predicted label | conf | displacement GT / pred / err (m) | rotation GT / pred / err (deg) |
|---|---|---|---|---|---|---|---|---|
| 1 | box of biscuits | added | missed | - | - | - |  |  |
| 2 | tea canister | removed | found | #3 removed | tea canister | 0.94 |  |  |
| 3 | white product box | moved | found | #1 moved | white product box | 0.92 | - / 0.79 / - | - / 54 / - |

Found 2/3 (type correct 2), missed 1, false positives 1.

### Extra predictions (false positives)

| id | type | label | conf |
|---|---|---|---|
| 2 | moved | silver trash can | 0.91 |

Unverified, not in the GT: #9 wooden stool (0.45), #10 black office chair (0.45), #11 blue suitcase (0.43), #12 wooden stool (0.40)


