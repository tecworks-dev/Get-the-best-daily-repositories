# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import random
import time
from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pyarrow as pa
import urllib3
from alibabacloud_credentials import providers
from odps import ODPS
from odps.accounts import AliyunAccount, BaseAccount, CredentialProviderAccount
from odps.apis.storage_api import (
    ArrowReader,
    ReadRowsRequest,
    SessionRequest,
    SessionStatus,
    SplitOptions,
    Status,
    StorageApiArrowClient,
    TableBatchScanRequest,
    TableBatchWriteRequest,
    WriteRowsRequest,
)
from odps.errors import ODPSError
from torch import distributed as dist

from tzrec.constant import Mode
from tzrec.datasets.dataset import BaseDataset, BaseReader, BaseWriter
from tzrec.features.feature import BaseFeature
from tzrec.protos import data_pb2
from tzrec.utils import dist_util

TYPE_TABLE_TO_PA = {
    "BIGINT": pa.int64(),
    "DOUBLE": pa.float64(),
    "FLOAT": pa.float32(),
    "STRING": pa.string(),
    "INT": pa.int32(),
    "ARRAY<BIGINT>": pa.list_(pa.int64()),
    "ARRAY<DOUBLE>": pa.list_(pa.float64()),
    "ARRAY<FLOAT>": pa.list_(pa.float32()),
    "ARRAY<STRING>": pa.list_(pa.string()),
    "ARRAY<INT>": pa.list_(pa.int32()),
    "ARRAY<ARRAY<BIGINT>>": pa.list_(pa.list_(pa.int64())),
    "ARRAY<ARRAY<DOUBLE>>": pa.list_(pa.list_(pa.float64())),
    "ARRAY<ARRAY<FLOAT>>": pa.list_(pa.list_(pa.float32())),
    "ARRAY<ARRAY<STRING>>": pa.list_(pa.list_(pa.string())),
    "ARRAY<ARRAY<INT>>": pa.list_(pa.list_(pa.int32())),
    "MAP<STRING,BIGINT>": pa.map_(pa.string(), pa.int64()),
    "MAP<STRING,DOUBLE>": pa.map_(pa.string(), pa.float64()),
    "MAP<STRING,FLOAT>": pa.map_(pa.string(), pa.float32()),
    "MAP<STRING,STRING>": pa.map_(pa.string(), pa.string()),
    "MAP<STRING,INT>": pa.map_(pa.string(), pa.int32()),
    "MAP<BIGINT,BIGINT>": pa.map_(pa.int64(), pa.int64()),
    "MAP<BIGINT,DOUBLE>": pa.map_(pa.int64(), pa.float64()),
    "MAP<BIGINT,FLOAT>": pa.map_(pa.int64(), pa.float32()),
    "MAP<BIGINT,STRING>": pa.map_(pa.int64(), pa.string()),
    "MAP<BIGINT,INT>": pa.map_(pa.int64(), pa.int32()),
    "MAP<INT,BIGINT>": pa.map_(pa.int32(), pa.int64()),
    "MAP<INT,DOUBLE>": pa.map_(pa.int32(), pa.float64()),
    "MAP<INT,FLOAT>": pa.map_(pa.int32(), pa.float32()),
    "MAP<INT,STRING>": pa.map_(pa.int32(), pa.string()),
    "MAP<INT,INT>": pa.map_(pa.int32(), pa.int32()),
}
TYPE_PA_TO_TABLE = {v: k for k, v in TYPE_TABLE_TO_PA.items()}


def _parse_odps_config_file(odps_config_path: str) -> Tuple[str, str, str]:
    """Parse odps config file."""
    if os.path.exists(odps_config_path):
        odps_config = {}
        with open(odps_config_path, "r") as f:
            for line in f.readlines():
                values = line.split("=", 1)
                if len(values) == 2:
                    odps_config[values[0]] = values[1].strip()
    else:
        raise ValueError("No such file: %s" % odps_config_path)

    try:
        os.environ["ACCESS_ID"] = odps_config["access_id"]
        os.environ["ACCESS_KEY"] = odps_config["access_key"]
        os.environ["ODPS_ENDPOINT"] = odps_config["end_point"]
    except KeyError as err:
        raise IOError(
            "%s key does not exist in the %s file." % (str(err), odps_config_path)
        ) from err

    return odps_config["access_id"], odps_config["access_key"], odps_config["end_point"]


def _create_odps_account() -> Tuple[BaseAccount, str]:
    account = None
    if "ODPS_CONFIG_FILE_PATH" in os.environ:
        account_id, account_key, odps_endpoint = _parse_odps_config_file(
            os.environ["ODPS_CONFIG_FILE_PATH"]
        )
        account = AliyunAccount(account_id, account_key)
    elif "ALIBABA_CLOUD_CREDENTIALS_URI" in os.environ:
        p = providers.DefaultCredentialsProvider()
        # prevent too much request to credential server after forked
        p.get_credentials().get_credential()
        account = CredentialProviderAccount(p)
        try:
            odps_endpoint = os.environ["ODPS_ENDPOINT"]
        except KeyError as err:
            raise RuntimeError(
                "ODPS_ENDPOINT does not exist in environment variables."
            ) from err
    else:
        account_id, account_key, odps_endpoint = _parse_odps_config_file(
            os.path.join(os.getenv("HOME", "/home/admin"), ".odps_config.ini")
        )
        account = AliyunAccount(account_id, account_key)

    return account, odps_endpoint


def _parse_table_path(odps_table_path: str) -> Tuple[str, str, Optional[List[str]]]:
    """Method that parse odps table path."""
    str_list = odps_table_path.split("/")
    if len(str_list) < 5 or str_list[3] != "tables":
        raise ValueError(
            f"'{odps_table_path}' is invalid, please refer:"
            "'odps://${your_projectname}/tables/${table_name}/${pt_1}/${pt_2}&${pt_1}/${pt_2}'"
        )

    table_partition = "/".join(str_list[5:])
    if not table_partition:
        table_partitions = None
    else:
        table_partitions = table_partition.split("&")
    return str_list[2], str_list[4], table_partitions


def _calc_slice_position(
    row_count: int,
    slice_id: int,
    slice_count: int,
    batch_size: int,
    drop_redundant_bs_eq_one: bool,
) -> Tuple[int, int]:
    """Calc table read position according to the slice information."""
    size = int(row_count / slice_count)
    split_point = row_count % slice_count
    if slice_id < split_point:
        start = slice_id * (size + 1)
        end = start + (size + 1)
    else:
        start = split_point * (size + 1) + (slice_id - split_point) * size
        end = start + size
    # when (end - start) % bz = 1 on some workers and
    # (end - start) % bz = 0 on other workers, train_eval will hang
    if (
        drop_redundant_bs_eq_one
        and split_point != 0
        and (end - start) % batch_size == 1
        and size % batch_size == 0
    ):
        end = end - 1
    return start, end


def _read_rows_arrow_with_retry(
    client: StorageApiArrowClient,
    read_req: ReadRowsRequest,
) -> ArrowReader:
    max_retry_count = 3
    retry_cnt = 0
    while True:
        try:
            reader = client.read_rows_arrow(read_req)
        except ODPSError as e:
            if retry_cnt >= max_retry_count:
                raise e
            retry_cnt += 1
            time.sleep(random.choice([5, 9, 12]))
            continue
        break
    return reader


def _reader_iter(
    client: StorageApiArrowClient,
    sess_reqs: List[SessionRequest],
    worker_id: int,
    num_workers: int,
    batch_size: int,
    drop_redundant_bs_eq_one: bool,
) -> Iterator[pa.RecordBatch]:
    for sess_req in sess_reqs:
        while True:
            scan_resp = client.get_read_session(sess_req)
            if scan_resp.session_status == SessionStatus.INIT:
                time.sleep(1)
                continue
            break
        start, end = _calc_slice_position(
            # pyre-ignore [6]
            scan_resp.record_count,
            worker_id,
            num_workers,
            batch_size,
            drop_redundant_bs_eq_one,
        )

        offset = 0
        retry_cnt = 0
        read_req = ReadRowsRequest(
            session_id=sess_req.session_id,
            row_index=start,
            row_count=end - start,
            max_batch_rows=min(batch_size, 20000),
        )
        reader = _read_rows_arrow_with_retry(client, read_req)
        max_retry_count = 5
        while True:
            try:
                read_data = reader.read()
            except urllib3.exceptions.HTTPError as e:
                if retry_cnt >= max_retry_count:
                    raise e
                retry_cnt += 1
                read_req = ReadRowsRequest(
                    session_id=sess_req.session_id,
                    row_index=start + offset,
                    row_count=end - start - offset,
                    max_batch_rows=min(batch_size, 20000),
                )
                reader = _read_rows_arrow_with_retry(client, read_req)
                continue
            if read_data is None:
                break
            else:
                retry_cnt = 0
                offset += len(read_data)
            yield read_data


class OdpsDataset(BaseDataset):
    """Dataset for reading data in Odps(Maxcompute).

    Args:
        data_config (DataConfig): an instance of DataConfig.
        features (list): list of features.
        input_path (str): data input path.
    """

    def __init__(
        self,
        data_config: data_pb2.DataConfig,
        features: List[BaseFeature],
        input_path: str,
        **kwargs: Any,
    ) -> None:
        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            assert (
                dist.is_initialized()
            ), "You should initialize distribute group first."
        super().__init__(data_config, features, input_path, **kwargs)
        # pyre-ignore [29]
        self._reader = OdpsReader(
            input_path,
            self._batch_size,
            list(self._selected_input_names) if self._selected_input_names else None,
            self._data_config.drop_remainder,
            is_orderby_partition=self._data_config.is_orderby_partition,
            quota_name=self._data_config.odps_data_quota_name,
            drop_redundant_bs_eq_one=self._mode != Mode.PREDICT,
        )
        self._init_input_fields()
        # prevent graph-learn parse odps config hang
        os.environ["END_POINT"] = os.environ["ODPS_ENDPOINT"]


class OdpsReader(BaseReader):
    """Odps(Maxcompute) reader class.

    Args:
        input_path (str): data input path.
        batch_size (int): batch size.
        selected_cols (list): selection column names.
        drop_remainder (bool): drop last batch less than batch_size.
        is_orderby_partition (bool): read data order by table partitions or not.
        quota_name (str): storage api quota name.
        drop_redundant_bs_eq_one (bool): drop last redundant batch with batch_size
            equal one to prevent train_eval hung.
    """

    def __init__(
        self,
        input_path: str,
        batch_size: int,
        selected_cols: Optional[List[str]] = None,
        drop_remainder: bool = False,
        is_orderby_partition: bool = False,
        quota_name: str = "pay-as-you-go",
        drop_redundant_bs_eq_one: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(input_path, batch_size, selected_cols, drop_remainder)
        self._is_orderby_partition = is_orderby_partition
        self._quota_name = quota_name
        self._drop_redundant_bs_eq_one = drop_redundant_bs_eq_one

        self._account, self._odps_endpoint = _create_odps_account()
        self._proj_to_o = {}
        self._table_to_cli = {}
        self._input_to_sess = {}
        self._init_client()

        self.schema = []
        self._ordered_cols = []
        _, table_name, _ = _parse_table_path(self._input_path.split(",")[0])
        table = self._table_to_cli[table_name].table
        for column in table.schema.simple_columns:
            if not self._selected_cols or column.name in self._selected_cols:
                column_type = str(column.type).upper()
                if column_type not in TYPE_TABLE_TO_PA:
                    raise ValueError(
                        f"column [{column.name}] with dtype {column.type} "
                        "is not supported now."
                    )
                self.schema.append(
                    pa.field(column.name, TYPE_TABLE_TO_PA[str(column.type).upper()])
                )
                self._ordered_cols.append(column.name)
        self._init_session()

    def _init_client(self) -> None:
        """Init storage api client."""
        for input_path in self._input_path.split(","):
            project, table_name, _ = _parse_table_path(input_path)
            if project not in self._proj_to_o:
                self._proj_to_o[project] = ODPS(
                    account=self._account,
                    project=project,
                    endpoint=self._odps_endpoint,
                )
            if table_name not in self._table_to_cli:
                o = self._proj_to_o[project]
                self._table_to_cli[table_name] = StorageApiArrowClient(
                    odps=o, table=o.get_table(table_name), quota_name=self._quota_name
                )

    def _init_session(self) -> None:
        """Init table scan session."""
        for input_path in self._input_path.split(","):
            session_ids = []
            _, table_name, partitions = _parse_table_path(input_path)
            client = self._table_to_cli[table_name]
            if self._is_orderby_partition and partitions is not None:
                splited_partitions = [[x] for x in partitions]
            else:
                splited_partitions = [partitions]
            for partitions in splited_partitions:
                if int(os.environ.get("RANK", 0)) == 0:
                    scan_req = TableBatchScanRequest(
                        split_options=SplitOptions(split_mode="RowOffset"),
                        required_data_columns=self._ordered_cols,
                        required_partitions=partitions,
                    )
                    scan_resp = client.create_read_session(scan_req)
                    session_ids.append(scan_resp.session_id)
                else:
                    session_ids.append(None)
            if dist.is_initialized():
                dist.broadcast_object_list(session_ids)
            self._input_to_sess[input_path] = [
                SessionRequest(session_id=x) for x in session_ids
            ]

    def _iter_one_table(
        self, input_path: str, worker_id: int = 0, num_workers: int = 1
    ) -> Iterator[Dict[str, pa.Array]]:
        _, table_name, _ = _parse_table_path(input_path)
        client = self._table_to_cli[table_name]

        sess_reqs = self._input_to_sess[input_path]
        iterator = _reader_iter(
            client,
            sess_reqs,
            worker_id,
            num_workers,
            self._batch_size,
            self._drop_redundant_bs_eq_one,
        )
        yield from self._arrow_reader_iter(iterator)

    def to_batches(
        self, worker_id: int = 0, num_workers: int = 1
    ) -> Iterator[Dict[str, pa.Array]]:
        """Get batch iterator."""
        for input_path in self._input_path.split(","):
            yield from self._iter_one_table(input_path, worker_id, num_workers)


class OdpsWriter(BaseWriter):
    """Odps(Maxcompute) writer class.

    Args:
        output_path (str): data output path.
        quota_name (str): storage api quota name.
    """

    def __init__(
        self, output_path: str, quota_name: str = "pay-as-you-go", **kwargs: Any
    ) -> None:
        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            assert (
                dist.is_initialized()
            ), "You should initialize distribute group first."
        super().__init__(output_path)
        self._account, self._odps_endpoint = _create_odps_account()
        self._quota_name = quota_name

        self._project, self._table_name, partitions = _parse_table_path(output_path)
        if partitions is None:
            self._partition_spec = None
        else:
            self._partition_spec = partitions[0]
        self._o = ODPS(
            account=self._account,
            project=self._project,
            endpoint=self._odps_endpoint,
        )
        self._client = None
        self._sess_req = None
        self._writer = None
        if self._o.exist_table(self._table_name):
            if int(os.environ.get("RANK", 0)) == 0:
                self._create_partition()
            else:
                self._wait_init_table()
            self._init_writer()
            self._lazy_inited = True

    def _create_table(self, output_dict: OrderedDict[str, pa.Array]) -> None:
        """Create output table."""
        schemas = []
        for k, v in output_dict.items():
            schemas.append(f"{k} {TYPE_PA_TO_TABLE[v.type]}")
        schema = ",".join(schemas)
        if self._partition_spec:
            pt_schemas = []
            for pt_spec in self._partition_spec.split("/"):
                pt_name = pt_spec.split("=")[0]
                pt_schemas.append(f"{pt_name} STRING")
            schema = (schema, ",".join(pt_schemas))
        self._o.create_table(
            self._table_name, schema, hints={"odps.sql.type.system.odps2": "true"}
        )

    def _create_partition(self) -> None:
        """Create output partition."""
        if self._partition_spec:
            t = self._o.get_table(self._table_name)
            partition_spec = self._partition_spec.replace("/", ",")
            if not t.exist_partition(partition_spec):
                t.create_partition(partition_spec, if_not_exists=True)

    def _init_writer(self) -> None:
        """Initialize table writer."""
        self._client = StorageApiArrowClient(
            odps=self._o,
            table=self._o.get_table(self._table_name),
            quota_name=self._quota_name,
        )
        session_id = None
        if int(os.environ.get("RANK", 0)) == 0:
            write_req = TableBatchWriteRequest(
                partition_spec=self._partition_spec, overwrite=True
            )
            write_resp = self._client.create_write_session(write_req)
            session_id = write_resp.session_id
            print(session_id)
        if dist.is_initialized():
            session_id = dist_util.broadcast_string(session_id)
        self._sess_req = SessionRequest(session_id=session_id)
        while True:
            sess_resp = self._client.get_write_session(self._sess_req)
            if sess_resp.session_status == SessionStatus.INIT:
                time.sleep(1)
                continue
            break
        row_req = WriteRowsRequest(
            session_id=sess_resp.session_id, block_number=int(os.environ.get("RANK", 0))
        )
        self._writer = self._client.write_rows_arrow(row_req)

    def _wait_init_table(self) -> None:
        """Wait table and partition ready."""
        while True:
            if not self._o.exist_table(self._table_name):
                time.sleep(1)
                continue
            t = self._o.get_table(self._table_name)
            if self._partition_spec:
                partition_spec = self._partition_spec.replace("/", ",")
                if not t.exist_partition(partition_spec):
                    time.sleep(1)
                    continue
            break

    def write(self, output_dict: OrderedDict[str, pa.Array]) -> None:
        """Write a batch of data."""
        if not self._lazy_inited:
            if int(os.environ.get("RANK", 0)) == 0:
                self._create_table(output_dict)
                self._create_partition()
            else:
                self._wait_init_table()
            self._init_writer()
            self._lazy_inited = True
        record_batch = pa.RecordBatch.from_arrays(
            list(output_dict.values()),
            list(output_dict.keys()),
        )
        self._writer.write(record_batch)

    def close(self) -> None:
        """Close and commit data."""
        if self._writer is not None:
            commit_msg, _ = self._writer.finish()
            if dist.is_initialized():
                commit_msgs = dist_util.gather_strings(commit_msg)
            else:
                commit_msgs = [commit_msg]
            if int(os.environ.get("RANK", 0)) == 0:
                resp = self._client.commit_write_session(self._sess_req, commit_msgs)
                while resp.status == Status.WAIT:
                    time.sleep(1)
                    resp = self._client.get_write_session(self._sess_req)
                if resp.session_status != SessionStatus.COMMITTED:
                    raise RuntimeError(
                        f"Fail to commit write session: {self._sess_req.session_id}"
                    )