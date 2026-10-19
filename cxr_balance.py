"""Class-balanced epochs for the few-box ChestX-Det settings, the same rule for every method: the training items with a
focal finding (Nodule, Mass, Calcification, Fracture) are repeated until they make up the share they have when every box
is used. With every box this changes nothing; with few boxes each epoch keeps the full-data class balance and length."""
import math

FULL_IMAGE_SHARE = 0.2644   # training images with a focal box when every box is used (720 of 2723)
FULL_ROW_SHARE = 0.1607     # the focal reader's positive address rows when every box is used (5215 of 32445)


def balanced(items, is_pos, share=FULL_IMAGE_SHARE):
    pos = [x for x in items if is_pos(x)]
    neg = [x for x in items if not is_pos(x)]
    if not pos or share <= 0:
        return list(items)
    n = max(len(pos), math.ceil(share * len(neg) / (1 - share)))
    return neg + [pos[i % len(pos)] for i in range(n)]
