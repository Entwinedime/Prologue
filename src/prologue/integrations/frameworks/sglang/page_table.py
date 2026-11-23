"""Describe the native prebuilt page-table ports independently of where they were observed."""


class PageTable:
    copy_operation = "pd_page_copy"
    assembly_operation = "pd_page_table_assembly"

    @classmethod
    def copy(cls, operand):
        begin, end = operand["source"]
        fields = ("source_dtype", "target_dtype", "source_device", "target_device")
        return {"phase": cls.copy_operation, "length": end - begin,
                **{field: operand[field] for field in fields}}

    @classmethod
    def assembly(cls, operands):
        return {"phase": cls.assembly_operation, "members": [
            {"begin": operand["source"][0], "copy": cls.copy(operand)} for operand in operands
        ]}
