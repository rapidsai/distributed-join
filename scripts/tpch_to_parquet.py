"""
Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os
import os.path

import pyarrow
import pyarrow.csv
import pyarrow.parquet

from typing import List, Dict

col_names: Dict[str, List[str]] = {
    "lineitem": [
        "L_ORDERKEY", "L_PARTKEY", "L_SUPPKEY", "L_LINENUMBER", "L_QUANTITY",
        "L_EXTENDEDPRICE", "L_DISCOUNT", "L_TAX", "L_RETURNFLAG", "L_LINESTATUS",
        "L_SHIPDATE", "L_COMMITDATE", "L_RECEIPTDATE", "L_SHIPINSTRUCT", "L_SHIPMODE",
        "L_COMMENT", "PLACEHOLDER"],
    "orders": [
        "O_ORDERKEY", "O_CUSTKEY", "O_ORDERSTATUS", "O_TOTALPRICE", "O_ORDERDATE",
        "O_ORDERPRIORITY", "O_CLERK", "O_SHIPPRIORITY", "O_COMMENT", "PLACEHOLDER"]
}


col_types: Dict[str, Dict[str, pyarrow.lib.DataType]] = {
    "lineitem": {
        "L_ORDERKEY": pyarrow.int64(),
        "L_PARTKEY": pyarrow.int64(),
        "L_SUPPKEY": pyarrow.int64(),
        "L_LINENUMBER": pyarrow.int32(),
        # Note: Decimal type with precision is not supported in cuDF at the moment
        # See: https://github.com/rapidsai/cudf/issues/6656
        # "L_QUANTITY": pyarrow.decimal128(12, 2),
        # "L_EXTENDEDPRICE": pyarrow.decimal128(12, 2),
        # "L_DISCOUNT": pyarrow.decimal128(12, 2),
        # "L_TAX": pyarrow.decimal128(12, 2),
        "L_RETURNFLAG": pyarrow.string(),
        "L_LINESTATUS": pyarrow.string(),
        # TODO: From TPC-H specification, it should be possible to represent dates as 32-bit
        # integers, but pyarrow currently does not support that, with the following error
        #    pyarrow.lib.ArrowNotImplementedError: CSV conversion to date32[day] is not supported
        # "L_SHIPDATE": pyarrow.date32(),
        # "L_COMMITDATE": pyarrow.date32(),
        # "L_RECEIPTDATE": pyarrow.date32(),
        "L_SHIPINSTRUCT": pyarrow.string(),
        "L_SHIPMODE": pyarrow.string(),
        "L_COMMENT": pyarrow.string()
    },
    "orders": {
        "O_ORDERKEY": pyarrow.int64(),
        "O_CUSTKEY": pyarrow.int64(),
        "O_ORDERSTATUS": pyarrow.string(),
        # "O_TOTALPRICE": pyarrow.decimal128(12, 2),
        # "O_ORDERDATE": pyarrow.date32(),
        "O_ORDERPRIORITY": pyarrow.string(),
        "O_CLERK": pyarrow.string(),
        "O_SHIPPRIORITY": pyarrow.int32(),
        "O_COMMENT": pyarrow.string()
    }
}


def tpch_to_parquet(path: str, prefix: str) -> None:
    # TODO: Process each input file in chunks.
    # Currently, this function loads each input file into memory before writing to disk in Parquet
    # format. This requires the memory large enough to hold the table. To get away this requirement,
    # we could process each input file in small chunks. However, when I implemented it, I got the
    # following error:
    #     TypeError: Cannot convert pyarrow.lib.Int64Array to pyarrow.lib.RecordBatch
    # when converting a chunk to a pyarrow table.

    input_paths: List[str] = []

    for filename in os.listdir(path):
        input_path = os.path.join(path, filename)
        if filename.startswith(prefix) and os.path.isfile(input_path) \
                and not filename.endswith(".parquet"):
            input_paths.append(input_path)

    for input_path in input_paths:
        input_table = pyarrow.csv.read_csv(
            input_path,
            read_options=pyarrow.csv.ReadOptions(
                use_threads=True,
                column_names=col_names[prefix],
                autogenerate_column_names=False),
            parse_options=pyarrow.csv.ParseOptions(delimiter="|"),
            convert_options=pyarrow.csv.ConvertOptions(
                include_columns=col_names[prefix][:-1],
                column_types=col_types[prefix]))

        parquet_writer = pyarrow.parquet.ParquetWriter(
            input_path + ".parquet", input_table.schema, compression="snappy")

        parquet_writer.write_table(input_table)


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "path", type=str, help="path to the directory containing generated TPC-H tables")
    arguments = argument_parser.parse_args()
    path: str = arguments.path

    tpch_to_parquet(path, "lineitem")
    tpch_to_parquet(path, "orders")


if __name__ == '__main__':
    main()
