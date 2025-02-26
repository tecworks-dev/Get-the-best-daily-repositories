import { historicalTickerCandleType } from "@/interface";

export interface TimeItem {
  label: string;
  value: string;
  time: number;
}

export const timeArr: TimeItem[] = [
  // { label: "1S", value: "1s", time: 1 },
  { label: "1", value: "1m", time: 60 },
  { label: "5", value: "5m", time: 300 },
  { label: "15", value: "15m", time: 900 },
  { label: "30", value: "30m", time: 1800 },
  { label: "60", value: "1h", time: 3600 },
  { label: "240", value: "4h", time: 14400 },
  { label: "360", value: "6h", time: 21600 },
  { label: "1D", value: "1d", time: 86400 },
];

export interface Bar {
  time: number;
  low: number;
  high: number;
  open: number;
  close: number;
  volume: number;
}
export const fillMissingBars = (
  data: historicalTickerCandleType[],
  from: number,
  to: number,
  resolution: string
): Bar[] => {
  const resolutionToSeconds = timeArr.find(
    (item) => item.label === resolution
  )?.time;

  if (!resolutionToSeconds) {
    throw new Error("Unsupported resolution");
  }

  const interval = resolutionToSeconds;
  const bars: Bar[] = [];

  try {
    let currentTime = data[0]?.time || from; // 如果 data 为空，则从 `from` 开始
    // 遍历时间范围，逐步填充数据
    while (currentTime < new Date().getTime() / 1000) {
      const existingBar = data.find((bar) => bar.time === currentTime);

      if (existingBar) {
        // 如果当前时间点有数据，直接使用它
        bars.push({
          time: existingBar.time * 1000,
          low: Number(existingBar.low),
          high: Number(existingBar.high),
          open: bars.length
            ? Number(bars[bars.length - 1].close)
            : existingBar.open,
          close: Number(existingBar.close),
          volume: existingBar.volume,
        });
      } else if (bars.length > 0) {
        // 如果数据缺失，使用上一条数据填充
        const lastBar = bars[bars.length - 1];
        const newData = data.find((bar) => bar.time === currentTime + interval);
        if (newData) {
          bars.push({
            time: currentTime * 1000,
            low: newData.open,
            high: lastBar.close,
            open: lastBar.close, // 新数据的 open 使用上一条的 close
            close: newData.open,
            volume: 0, // 缺失数据时 volume 为 0
          });
        } else {
          bars.push({
            time: currentTime * 1000,
            low: lastBar.close,
            high: lastBar.close,
            open: lastBar.close, // 新数据的 open 使用上一条的 close
            close: lastBar.close,
            volume: 0, // 缺失数据时 volume 为 0
          });
        }
      }
      // 移动到下一个时间点
      currentTime += interval;
    }
  } catch (e) {
    return [];
  }

  return bars;
};

interface PeriodParams {
  from: number;
  to: number;
}

interface DatafeedConfiguration {
  supported_resolutions: string[];
  exchanges: string[];
  symbols_types: string[];
}

export const getDatafeed = (
  requireFun: (data: {
    internal: string;
    from: number;
    to: number;
  }) => Promise<historicalTickerCandleType[]>,
  socketFun: (value: string, onRealtimeCallback: Function) => void
) => {
  return {
    onReady: (callback: (config: DatafeedConfiguration) => void) => {
      const configurationData: DatafeedConfiguration = {
        supported_resolutions: timeArr.map((item) => item.label),
        exchanges: [],
        symbols_types: [],
      };
      setTimeout(() => callback(configurationData));
    },
    searchSymbols: (
      userInput: string,
      exchange: string,
      symbolType: string,
      onResultReadyCallback: (result: any[]) => void
    ) => {
      console.log("[searchSymbols]: Method call");
    },
    resolveSymbol: (
      symbolName: string,
      onSymbolResolvedCallback: (symbolInfo: any) => void,
      onResolveErrorCallback: (error: string) => void,
      extension: any
    ) => {
      const symbolInfo = {
        name: "",
        ticker: "",
        id: 1,
        session: "0000-2359:1234567",
        has_daily: true, //是否具有日K历史数据
        has_seconds: true, //是否具有分时K历史数据
        has_intraday: true, //是否具有分钟K历史数据
        has_no_volume: false, //默认关闭成交量
        visible_plots_set: false,
        has_weekly_and_monthly: true,
        minmov: 1,
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        pricescale: Math.pow(10, 6),
        supports_time: true,
        supported_resolutions: timeArr.map((item) => item.label),
      };
      setTimeout(() => onSymbolResolvedCallback(symbolInfo));
    },
    getBars: async (
      symbolInfo: any,
      resolution: string,
      periodParams: PeriodParams,
      onHistoryCallback: (bars: Bar[], meta: { noData: boolean }) => void,
      onErrorCallback: (error: string) => void
    ) => {
      const { from, to } = periodParams;
      try {
        const requireData = {
          internal:
            timeArr.find((item) => item.label === resolution)?.value || "",
          from,
          to,
        };
        const data = await requireFun(requireData);
        const bars = fillMissingBars(data, from, to, resolution);
        if (bars.length >= 1000) {
          onHistoryCallback(bars, { noData: false });
        } else {
          onHistoryCallback(bars, { noData: true });
        }
      } catch (e) {
        console.log("Error in getBars:", e);
        onErrorCallback("Error fetching data");
      }
    },
    subscribeBars: (
      symbolInfo: any,
      resolution: string,
      onRealtimeCallback: (bar: Bar) => void,
      subscriberUID: string,
      onResetCacheNeededCallback: () => void
    ) => {
      const result = subscriberUID.slice(4);
      const { value } = timeArr.find((item) => item.label === result) || {};
      if (value) {
        socketFun(value, onRealtimeCallback);
      }
    },
    unsubscribeBars: (subscriberUID: string) => {
      console.log(
        "[unsubscribeBars]: Method call with subscriberUID:",
        subscriberUID
      );
    },
  };
};

export default getDatafeed;
