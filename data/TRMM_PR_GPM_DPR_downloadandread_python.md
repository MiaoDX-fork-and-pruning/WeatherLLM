# TRMM PR 和 GPM DPR 雷达数据下载与 Python 读取说明

本文档说明如何注册 NASA Earthdata 账号、下载 TRMM PR 与 GPM DPR 二级雷达产品，以及如何用 Python 读取 HDF5 数据文件。示例以 GES DISC 在线目录为入口：

- TRMM PR：<https://disc2.gesdisc.eosdis.nasa.gov/data/TRMM_L2/GPM_2APR.07/>
- GPM DPR：<https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L2/GPM_2ADPR.07/>

## 1. 数据与目录说明

| 传感器 | 产品目录 | 常见文件类型 | 说明 |
| --- | --- | --- | --- |
| TRMM PR | `TRMM_L2/GPM_2APR.07` | `.HDF5` | TRMM Precipitation Radar 二级降水雷达产品，V07 版本 |
| GPM DPR | `GPM_L2/GPM_2ADPR.07` | `.HDF5` | GPM Dual-frequency Precipitation Radar 二级降水雷达产品，V07 版本 |

GES DISC 目录通常按 `年/年积日` 组织，例如 `2020/001/` 表示 2020 年第 1 天。进入对应日期目录后，可下载单轨道文件。实际目录层级和文件名以网页显示为准。

常见文件名大致类似：

```text
2A.TRMM.PR...V07*.HDF5
2A.GPM.DPR...V07*.HDF5
```

## 2. 注册 Earthdata 账号并授权 GES DISC

1. 打开 Earthdata Login：<https://urs.earthdata.nasa.gov/>
2. 点击注册账号，填写用户名、邮箱、密码和用途等信息。
3. 通过邮件完成账号激活。
4. 登录后，打开任一 GES DISC 数据目录，例如：
   - <https://disc2.gesdisc.eosdis.nasa.gov/data/TRMM_L2/GPM_2APR.07/>
   - <https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L2/GPM_2ADPR.07/>
5. 第一次访问受保护数据时，页面可能提示授权应用。允许 `NASA GESDISC DATA ARCHIVE` 访问你的 Earthdata 账号。

如果命令行下载时一直返回登录页或 401/403 错误，通常是以下原因：

- Earthdata 账号未激活。
- 尚未授权 `NASA GESDISC DATA ARCHIVE`。
- `.netrc` 中用户名或密码错误。
- 下载命令没有使用 cookie 或没有跟随重定向。

## 3. 配置命令行认证

建议把 Earthdata 账号写入用户主目录下的 `.netrc` 文件，这样 `curl`、`wget` 和 Python `requests` 都能复用认证信息。

### Windows PowerShell

在 PowerShell 中执行：

```powershell
notepad $HOME\.netrc
```

写入以下内容，并替换为自己的 Earthdata 用户名和密码：

```text
machine urs.earthdata.nasa.gov login YOUR_USERNAME password YOUR_PASSWORD
```

然后创建 cookie 文件：

```powershell
New-Item -Path $HOME\.urs_cookies -ItemType File -Force
```

在 PowerShell 中建议使用 `curl.exe`，避免调用到 PowerShell 的 `curl` 别名。

### Linux 或 macOS

```bash
cat > ~/.netrc <<'EOF'
machine urs.earthdata.nasa.gov login YOUR_USERNAME password YOUR_PASSWORD
EOF

chmod 600 ~/.netrc
touch ~/.urs_cookies
```

## 4. 下载数据

### 4.1 浏览器手动下载

1. 打开产品目录。
2. 进入年份目录，例如 `2020/`。
3. 进入年积日目录，例如 `001/`。
4. 点击需要的 `.HDF5` 文件。
5. 如果浏览器要求登录，使用 Earthdata 账号登录并授权 GES DISC。

适合少量文件下载；批量下载建议使用 `curl`、`wget` 或 Python。

### 4.2 使用 curl 下载单个文件

把 `FILE_URL` 替换为实际文件链接。

Windows PowerShell：

```powershell
curl.exe -L -n -b "$HOME\.urs_cookies" -c "$HOME\.urs_cookies" -OJ "FILE_URL"
```

Linux 或 macOS：

```bash
curl -L -n -b ~/.urs_cookies -c ~/.urs_cookies -OJ "FILE_URL"
```

参数含义：

- `-L`：跟随 Earthdata 登录重定向。
- `-n`：读取 `.netrc` 中的用户名和密码。
- `-b` / `-c`：读取和保存 cookie。
- `-O`：使用远程文件名保存。
- `-J`：允许服务器提供下载文件名。

### 4.3 使用 wget 下载单个文件

```bash
wget --load-cookies ~/.urs_cookies \
     --save-cookies ~/.urs_cookies \
     --keep-session-cookies \
     --auth-no-challenge=on \
     --content-disposition \
     "FILE_URL"
```

Windows 也可使用 wget，但需要先安装，例如通过 Git Bash、MSYS2、conda 或系统包管理器。

### 4.4 批量下载 URL 列表

先创建 `urls.txt`，每行一个文件 URL，例如：

```text
https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L2/GPM_2ADPR.07/2020/001/xxxxxxxx.HDF5
https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L2/GPM_2ADPR.07/2020/002/yyyyyyyy.HDF5
```

Windows PowerShell：

```powershell
New-Item -ItemType Directory -Force -Path data\gpm_dpr
Get-Content .\urls.txt | ForEach-Object {
    if ($_ -and -not $_.StartsWith("#")) {
        curl.exe -L -n -b "$HOME\.urs_cookies" -c "$HOME\.urs_cookies" -o ("data\gpm_dpr\" + [System.IO.Path]::GetFileName($_)) $_
    }
}
```

Linux 或 macOS：

```bash
mkdir -p data/gpm_dpr
while read -r url; do
    [ -z "$url" ] && continue
    case "$url" in \#*) continue ;; esac
    curl -L -n -b ~/.urs_cookies -c ~/.urs_cookies -o "data/gpm_dpr/$(basename "$url")" "$url"
done < urls.txt
```

## 5. Python 环境

建议使用独立环境：

```bash
conda create -n radar_hdf5 python=3.11 -y
conda activate radar_hdf5
pip install h5py numpy matplotlib requests tqdm
```

核心读取依赖是 `h5py` 和 `numpy`；`matplotlib` 用于快速画图；`requests` 和 `tqdm` 用于 Python 批量下载。

## 6. Python 批量下载示例

下面脚本会读取 `urls.txt`，把文件下载到指定目录。它优先使用 `.netrc` 中的 Earthdata 账号，不需要把密码写入脚本。

```python
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm


class SessionWithHeaderRedirection(requests.Session):
    """Keep Earthdata authorization during redirects to urs.earthdata.nasa.gov."""

    AUTH_HOST = "urs.earthdata.nasa.gov"

    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        if "Authorization" not in headers:
            return

        original = urlparse(response.request.url)
        redirect = urlparse(prepared_request.url)

        if original.hostname != redirect.hostname and redirect.hostname != self.AUTH_HOST:
            del headers["Authorization"]


def download_file(url: str, out_dir: Path, session: requests.Session) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(url).path).name
    out_path = out_dir / filename

    with session.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with out_path.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=filename
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    return out_path


def main():
    url_file = Path("urls.txt")
    out_dir = Path("data/gpm_dpr")

    session = SessionWithHeaderRedirection()
    session.trust_env = True  # allows requests to read ~/.netrc

    urls = [
        line.strip()
        for line in url_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    for url in urls:
        path = download_file(url, out_dir, session)
        print(f"Downloaded: {path}")


if __name__ == "__main__":
    main()
```

如果运行后保存的是 HTML 登录页而不是 `.HDF5` 文件，说明认证或授权没有生效。先用浏览器打开同一个文件链接，确认能够登录并授权 GES DISC。

## 7. 查看 HDF5 文件结构

TRMM PR 和 GPM DPR 的 HDF5 文件包含多个 group 和 dataset。不同产品、版本、扫描模式的变量路径可能略有差异，因此建议先打印文件结构。

```python
from pathlib import Path

import h5py


def print_hdf5_tree(file_path: str | Path, max_items: int = 300):
    file_path = Path(file_path)
    count = 0

    with h5py.File(file_path, "r") as h5:
        def visitor(name, obj):
            nonlocal count
            if count >= max_items:
                return

            if isinstance(obj, h5py.Dataset):
                print(f"{name:70s} shape={obj.shape} dtype={obj.dtype}")
                count += 1

        h5.visititems(visitor)


print_hdf5_tree("data/gpm_dpr/example.HDF5")
```

## 8. 读取近地表降水率

下面示例读取经纬度和近地表降水率，并自动处理常见的缺测值、比例因子和偏移量属性。

```python
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _scalar_attr(attrs, names, default=None):
    for name in names:
        if name in attrs:
            value = attrs[name]
            array = np.asarray(value)
            if array.size == 1:
                return array.item()
            return value
    return default


def read_dataset(h5: h5py.File, path: str) -> np.ndarray:
    dataset = h5[path]
    data = dataset[()]

    if np.issubdtype(data.dtype, np.number):
        data = data.astype(np.float32)

        fill_value = _scalar_attr(dataset.attrs, ["_FillValue", "MissingValue", "missing_value"])
        if fill_value is not None:
            data = np.where(data == fill_value, np.nan, data)

        scale = _scalar_attr(dataset.attrs, ["scale_factor", "ScaleFactor"], 1.0)
        offset = _scalar_attr(dataset.attrs, ["add_offset", "Offset"], 0.0)
        data = data * float(scale) + float(offset)

    return data


def first_existing_path(h5: h5py.File, candidates: list[str]) -> str:
    for path in candidates:
        if path in h5:
            return path
    raise KeyError(f"None of these paths exist: {candidates}")


def read_near_surface_precip(file_path: str | Path, swath: str = "FS"):
    file_path = Path(file_path)

    with h5py.File(file_path, "r") as h5:
        lat = read_dataset(h5, f"{swath}/Latitude")
        lon = read_dataset(h5, f"{swath}/Longitude")
        rain_path = first_existing_path(
            h5,
            [
                f"{swath}/SLV/precipRateNearSurface",
                f"{swath}/SLV/precipRateNearSurfaceLiquid",
                f"{swath}/SLV/precipRateESurface",
            ],
        )
        rain = read_dataset(h5, rain_path)

    return lon, lat, rain


file_path = "data/gpm_dpr/example.HDF5"
lon, lat, rain = read_near_surface_precip(file_path, swath="FS")

print(lon.shape, lat.shape, rain.shape)
print("rain min/max:", np.nanmin(rain), np.nanmax(rain))

plt.figure(figsize=(8, 5))
plt.scatter(lon, lat, c=rain, s=1, cmap="turbo", vmin=0, vmax=50)
plt.colorbar(label="Near-surface precipitation rate (mm h-1)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Near-surface precipitation")
plt.tight_layout()
plt.show()
```


## 9. 读取三维降水廓线

`precipRate` 通常是三维数组，维度一般为：

```text
扫描线数量 × 波束数量 × 垂直层数量
```

下面示例读取降水率垂直廓线，并绘制中间波束的沿轨剖面。

```python
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


file_path = Path("data/gpm_dpr/example.HDF5")
swath = "FS"

with h5py.File(file_path, "r") as h5:
    precip_profile = read_dataset(h5, f"{swath}/SLV/precipRate")
    lat = read_dataset(h5, f"{swath}/Latitude")

print("precip profile shape:", precip_profile.shape)

middle_ray = precip_profile.shape[1] // 2
curtain = precip_profile[:, middle_ray, :]

plt.figure(figsize=(10, 4))
plt.imshow(
    curtain.T,
    origin="lower",
    aspect="auto",
    cmap="turbo",
    vmin=0,
    vmax=50,
)
plt.colorbar(label="Precipitation rate (mm h-1)")
plt.xlabel("Scan index")
plt.ylabel("Vertical bin")
plt.title(f"{swath} middle-ray precipitation profile")
plt.tight_layout()
plt.show()
```

垂直层的高度需要结合产品中的高度、层号或官方文件说明解释。不同扫描模式和产品版本的垂直坐标定义可能不同，做定量廓线分析前应核对产品说明文档。

## 10. 读取扫描时间

扫描时间通常存放在 `ScanTime` 组中。

```python
from datetime import datetime

import h5py


def read_scan_time(h5: h5py.File, swath: str = "FS") -> list[datetime]:
    group = f"{swath}/ScanTime"
    year = h5[f"{group}/Year"][:]
    month = h5[f"{group}/Month"][:]
    day = h5[f"{group}/DayOfMonth"][:]
    hour = h5[f"{group}/Hour"][:]
    minute = h5[f"{group}/Minute"][:]
    second = h5[f"{group}/Second"][:]

    return [
        datetime(int(y), int(m), int(d), int(h), int(mi), int(s))
        for y, m, d, h, mi, s in zip(year, month, day, hour, minute, second)
    ]


with h5py.File("data/gpm_dpr/example.HDF5", "r") as h5:
    scan_times = read_scan_time(h5, swath="FS")

print(scan_times[0], scan_times[-1])
```

时间一般为 UTC。若要转换为北京时间，可在后续处理中加 8 小时或使用 `zoneinfo` 进行时区转换。


## 11. 参考链接

- Earthdata Login：<https://urs.earthdata.nasa.gov/>
- GES DISC 数据访问说明：<https://disc.gsfc.nasa.gov/data-access>
- Earthdata 命令行与脚本访问说明：<https://urs.earthdata.nasa.gov/documentation/for_users/data_access>
- GPM 数据目录：<https://gpm.nasa.gov/data/directory>
- TRMM PR GES DISC 目录：<https://disc2.gesdisc.eosdis.nasa.gov/data/TRMM_L2/GPM_2APR.07/>
- GPM DPR GES DISC 目录：<https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L2/GPM_2ADPR.07/>
