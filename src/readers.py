from pathlib import Path
from typing import Dict, List, Optional, Union, get_args, get_origin, get_type_hints
import re
from pyssian import GaussianInFile

from .main import autoVBInputData, VBSettings, SUPPORTED_METHODS, XMIPassthrough
from .utils import print_warning, print_subroutine

class autoVBInputParser:
    '''
    解析输入文件，提取必要的信息，如分子结构、基组、计算参数等
    '''
    def __init__(self, input_path: Path):
        self.input_path = input_path
        self.text = self.input_path.read_text(errors='ignore')
        # vbscf(2,1)会覆盖VBSettings中的nae/nao值，因此需要单独存储
        self.cmd_nae: int | None = None
        self.cmd_nao: int | None = None
        suffix = self.input_path.suffix.lower()

        # 检查是否使用 .xmi 格式输入
        if ('$ctrl' in self.text.lower() and '$geo' in self.text.lower()) or suffix == ".xmi":
            print('Detected .xmi input file format...')
            is_xmi = True
        else:
            is_xmi = False

        print(f"Content of autoVB input file {self.input_path}:")
        print_subroutine(self.text)

        if is_xmi:
            self.input_data = self.parse_xmi()
        else:
            self.input_data = self.parse_gaussian()
            settings = self.parse_autovb_options(self.input_data.title)

            # 如果 method 中通过 vbscf(nae,nao) 提供了显式值，则覆盖 settings
            if self.cmd_nae is not None or self.cmd_nao is not None:
                # warn if settings already specified nao/nae
                if getattr(settings, 'nae', 0) and getattr(settings, 'nao', 0):
                    print_warning(f"VBSettings in input file contains 'nae' and 'nao' but method provided overrides; using method values {self.cmd_nae},{self.cmd_nao} and ignoring commandline values.")
                if self.cmd_nae is not None:
                    settings.nae = int(self.cmd_nae)
                if self.cmd_nao is not None:
                    settings.nao = int(self.cmd_nao)

            self.input_data.vbsettings = settings

    def parse_xmi(self) -> autoVBInputData:

        lines = self.text.splitlines()
        title = self.input_path.stem
        for ln in lines:
            if ln.strip():
                title = ln.strip()
                break

        # 将 xmi 拆分为各个 section（$ctrl/$geo/$actorb/$str 等）
        sections: Dict[str, List[List[str]]] = {}
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if not stripped.startswith("$"):
                i += 1
                continue
            sec_name = stripped[1:].strip().lower()
            if sec_name == "end":
                i += 1
                continue
            i += 1
            body_lines: List[str] = []
            while i < len(lines):
                cur = lines[i].strip().lower()
                if cur == "$end":
                    break
                body_lines.append(lines[i])
                i += 1
            sections.setdefault(sec_name, []).append(body_lines)
            # 跳过 $end
            while i < len(lines) and lines[i].strip().lower() != "$end":
                i += 1
            if i < len(lines) and lines[i].strip().lower() == "$end":
                i += 1

        settings = VBSettings()
        type_hints = get_type_hints(VBSettings)
        alias_map = {"int": "inte", "str": "stru"}
        passthrough = XMIPassthrough()

        # 默认方法
        method: Optional[str] = None
        basis: Optional[str] = None

        ctrl_lines = sections.get("ctrl", [[]])[0] if sections.get("ctrl") else []
        for raw_line in ctrl_lines:
            s = raw_line.strip()
            if not s:
                continue

            if "=" in s:
                key, value = s.split("=", 1)
                key = key.strip()
                value = value.strip()
                key_lower = key.lower()
                mapped_key = alias_map.get(key_lower, key_lower)

                if mapped_key == "basis":
                    basis = value
                    continue

                if hasattr(settings, mapped_key):
                    target_type = type_hints.get(mapped_key, str)
                    try:
                        parsed_value = self.parse_value_by_type(value, target_type, mapped_key)
                        setattr(settings, mapped_key, parsed_value)
                    except Exception as e:
                        print(f"Warning: failed to parse $ctrl key '{key}' with value '{value}': {e}")
                    continue

                passthrough.ctrl_extra_lines.append(raw_line)
                continue

            # 无等号的开关型字段，例如 sort
            key_lower = s.lower()
            if key_lower in SUPPORTED_METHODS:
                if method is None:
                    method = key_lower
                continue
            mapped_key = alias_map.get(key_lower, key_lower)
            if hasattr(settings, mapped_key):
                target_type = type_hints.get(mapped_key, bool)
                try:
                    parsed_value = self.parse_value_by_type(True, target_type, mapped_key)
                    setattr(settings, mapped_key, parsed_value)
                except Exception as e:
                    print(f"Warning: failed to parse flag field '{s}' in $ctrl: {e}")
            else:
                passthrough.ctrl_extra_lines.append(raw_line)

        if method is None:
            raise ValueError(f"Failed to parse method from $ctrl section in {self.input_path}")
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method: {method}. Supported methods are: {SUPPORTED_METHODS}")
        if not basis:
            raise ValueError(f"Failed to parse basis from $ctrl section in {self.input_path}")

        # $actorb 等价于 aoa
        actorb_blocks = sections.get("actorb", [])
        if actorb_blocks:
            actorb_text = "\n".join(actorb_blocks[0])
            actorb_numbers = re.findall(r"[+-]?\d+", actorb_text)
            settings.aoa = [int(x) for x in actorb_numbers]

        # $str 原样透传
        str_blocks = sections.get("str", [])
        if str_blocks:
            passthrough.str_section_text = "\n".join(str_blocks[0])

        # 解析几何
        geo_lines = sections.get("geo", [[]])[0] if sections.get("geo") else []
        if not geo_lines:
            raise ValueError(f"Failed to parse $geo section from {self.input_path}")
        geometry = "\n".join(geo_lines).strip()

        settings.validate()

        atvb_input = autoVBInputData(
            title=title,
            filepath=self.input_path,
            filename=self.input_path.stem,
            method=method,
            basis=basis,
            geometry=geometry,
            vbsettings=settings,
            xmi_passthrough=passthrough,
        )
        print(f"Parsed XMI input file {self.input_path} successfully with method {method} and basis {basis}")
        return atvb_input

    def parse_gaussian(self) -> autoVBInputData:
        with GaussianInFile(self.input_path) as input_file:
            input_file.read()
        # method和basis不会自动读取，原因是GaussianInFile不支持VB方法的读取，它会识别成一整个参数
        # 识别包含VB或/的行，提取method和basis
        cmd_line = input_file.commandline
        method = None
        basis = None
        for i in cmd_line.items():
            key: str = i[0]
            if "/" in key:
                method_basis = key.split('/')
                method_raw = method_basis[0].strip()
                basis = method_basis[1]

                # 解析形如 vbscf(2,1) 的结构
                m = re.fullmatch(r"([A-Za-z0-9_+-]+)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", method_raw)
                if m:
                    method = m.group(1).lower()
                    try:
                        self.cmd_nae = int(m.group(2))
                        self.cmd_nao = int(m.group(3))
                    except Exception:
                        raise ValueError(f"Failed to parse nae/nao from method specification: {method_raw}")
                else:
                    method = method_raw.lower()

                if method not in SUPPORTED_METHODS:
                    raise ValueError(f"Unsupported method: {method}. Supported methods are: {SUPPORTED_METHODS}")
                break
        if not method or not basis:
            raise ValueError(f"Failed to parse method and basis from input file {self.input_path}. Ensure that the method and basis in the format 'method/basis'.")
        atvb_input = autoVBInputData(
            title=input_file.title,
            filepath=self.input_path,
            filename=self.input_path.stem,
            method=method,
            basis=basis,
            geometry=input_file.geometry,
            charge=input_file.charge,
            spin=input_file.spin,
            mem=input_file.mem,
            nproc=input_file.nprocs,
        )
        print(f"Parsed input file {self.input_path} successfully with method {method} and basis {basis}")

        return atvb_input
    
    def parse_value_by_type(self, raw: str, target_type, key: str):
        if isinstance(raw, str):
            raw = raw.strip()
        origin = get_origin(target_type)
        # print(f"Parsing value for key '{key}': raw='{raw}', target_type={target_type}, origin={origin}")
        if get_origin(target_type) == Union:
            actual_types = get_args(target_type)
            target_type = actual_types[0]

        if isinstance(raw, bool) and target_type is bool:
            return raw
        # 布尔判断（支持 1/0/true/false/yes/no）
        if target_type is bool:
            return raw.lower() in ("1", "true", "yes", "y")
        # 整数
        if target_type is int:
            try:
                return int(raw)
            except Exception:
                return int(float(raw))
        # 浮点
        if target_type is float:
            return float(raw)
        # 列表（尝试解析数字或字符串列表）
        if origin is list or target_type is list:
            s = raw
            # 去除括号/中括号
            if s.startswith(("(", "[")) and s.endswith((")", "]")):
                s = s[1:-1]
            # 用分号或空白分割
            parts = [p for p in re.split(r'[;,\s]+', s) if p != ""]
            parsed = []
            for p in parts:
                if re.fullmatch(r'[+-]?\d+', p):
                    parsed.append(int(p))
                else:
                    try:
                        parsed.append(float(p))
                    except Exception:
                        parsed.append(p)
            if key == "aoa_bond":
                if all(isinstance(x, int) for x in parsed):
                    return [parsed[i : i + 2] for i in range(0, len(parsed), 2)]
            return parsed
        if target_type is Path:
            return Path(raw)
        # 默认当字符串返回（去掉外层单/双引号）
        if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
            return raw[1:-1]
        return raw

    def parse_autovb_options(self, s: str) -> VBSettings:
        """
        解析形如: autovb{nae=4, nao=4}的字符串
        提取大括号内部的键值对，并把识别到的字段注入到 VBSettings dataclass 中。
        - 只注入 VBSettings 中已声明的字段，类型会尝试转换（int/float/bool/list/str）。
        - 未识别或非 VBSettings 字段会被忽略。
        """
        m = re.search(r"\{([^}]*)\}", s, re.DOTALL)
        if not m:
            return VBSettings()  # 无选项，返回默认

        inner = m.group(1)
        pattern = r'\s*(?:\([^()]*\)|[^,])+\s*'
        pair_list: list[str] = [p.strip() for p in re.findall(pattern, inner) if p.strip()]
        settings = VBSettings()
        type_hints = get_type_hints(VBSettings)
        alias_map = {"int": "inte", "str": "stru"}

        for pair in pair_list:
            if "=" not in pair:
                key = pair.strip()
                value = True
            else:
                key, value = pair.split("=", 1)
                key = key.strip()
                value = value.strip(' ')

            key = alias_map.get(key.lower(), key)
            if not hasattr(settings, key):
                continue
            target_type = type_hints.get(key, str)
            try:
                parsed_value = self.parse_value_by_type(value, target_type, key)
                setattr(settings, key, parsed_value)
                # print(f"Set VBSettings.{key} = {parsed_value} (parsed from '{value}')")
            except Exception as e:
                print(f"Warning: failed to parse value for key '{key}' with raw value '{value}'. Error: {e}. Skipping this option.")
                continue

        # 验证 VBSettings 合法性（若不合法会抛错并中止流程）
        settings.validate()

        return settings

class GaussianNBOParser:
    '''
    解析 Gaussian 的 NBO 输出文件，提取必要的信息，如 NBO 分析结果等
    '''
    def __init__(self, nbo_output_path: Path):
        self.nbo_output_path = nbo_output_path

    def parse(self):
        # 这里实现 NBO 输出文件的解析逻辑，提取所需信息
        pass