from typing import Optional, List
import ctypes, subprocess, pathlib, tempfile
from toonygrad.device import Compiled, Compiler, MallocAllocator
from toonygrad.helpers import cpu_time_execution, DEBUG, cpu_objdump
from toonygrad.renderer.cstyle import ClangRenderer

class ClangCompiler(Compiler):
  def __init__(self, cachekey="compile_clang", args:Optional[List[str]]=None):
    self.args = ['-march=native'] if args is None else args
    super().__init__(cachekey)

  def compile(self, src:str) -> bytes:
    # TODO: remove file write. sadly clang doesn't like the use of /dev/stdout here
    with tempfile.NamedTemporaryFile(delete=True) as output_file:
      subprocess.check_output(['clang', '-shared', *self.args, '-O2', '-Wall', '-Werror', '-x', 'c', '-fPIC', '-ffreestanding', '-nostdlib',
                               '-', '-o', str(output_file.name)], input=src.encode('utf-8'))
      return pathlib.Path(output_file.name).read_bytes()

class ClangProgram:
  def __init__(self, name:str, lib:bytes):
    if DEBUG >= 6: cpu_objdump(lib)
    self.name, self.lib = name, lib
    # write to disk so we can load it
    with tempfile.NamedTemporaryFile(delete=True) as cached_file_path:
      pathlib.Path(cached_file_path.name).write_bytes(lib)
      self.fxn = ctypes.CDLL(str(cached_file_path.name))[name]

  def __call__(self, *bufs, vals=(), wait=False): return cpu_time_execution(lambda: self.fxn(*bufs, *vals), enable=wait)

class ClangDevice(Compiled):
  def __init__(self, device:str):
    from toonygrad.runtime.graph.clang import ClangGraph
    super().__init__(device, MallocAllocator, ClangRenderer(), ClangCompiler(), ClangProgram, ClangGraph)
