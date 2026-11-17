"""
Retail Shelf Planogram Compliance with Open-Set SKU Recognition
================================================================

CSE437 Computer Vision Project - Daffodil International University

This package implements an end-to-end AI pipeline for auditing
supermarket shelf planogram compliance using:
    - Class-agnostic product detection (YOLOv8 + SKU-110K)
    - Open-set SKU recognition via embedding retrieval (F2/F3)
    - Planogram alignment via sequence edit distance (F4)
    - Violation injection for synthetic test generation (F5)
    - Dynamic gallery extension without retraining (F6)
"""

__version__ = "0.1.0"
__project__ = "Retail Shelf Planogram Compliance"
__course__ = "CSE437 - Computer Vision"
