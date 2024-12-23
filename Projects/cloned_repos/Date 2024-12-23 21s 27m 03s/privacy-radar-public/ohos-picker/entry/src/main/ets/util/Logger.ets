/*
 * Copyright (c) 2022 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import hilog from '@ohos.hilog';

/**
 * Common log for all features.
 *
 * @param {string} prefix Identifies the log tag.
 */
export class Logger {
  private domain: number;
  private prefix: string;
  private format: string = `%{public}s`;
  private format2Val: string = `%{public}s, %{public}s`;

  constructor(prefix: string) {
    this.prefix = prefix;
    this.domain = 0xFF00;
  }

  debug(...args: string[]) {
    if (args.length > 1) {
      hilog.debug(this.domain, this.prefix, this.format2Val, args);
    } else {
      hilog.debug(this.domain, this.prefix, this.format, args);
    }
  }

  info(...args: string[]) {
    if (args.length > 1) {
      hilog.info(this.domain, this.prefix, this.format2Val, args);
    } else {
      hilog.info(this.domain, this.prefix, this.format, args);
    }
  }

  warn(...args: string[]) {
    if (args.length > 1) {
      hilog.warn(this.domain, this.prefix, this.format2Val, args);
    } else {
      hilog.warn(this.domain, this.prefix, this.format, args);
    }
  }

  error(...args: string[]) {
    if (args.length > 1) {
      hilog.error(this.domain, this.prefix, this.format2Val, args);
    } else {
      hilog.error(this.domain, this.prefix, this.format, args);
    }
  }

  fatal(...args: string[]) {
    if (args.length > 1) {
      hilog.fatal(this.domain, this.prefix, this.format2Val, args);
    } else {
      hilog.fatal(this.domain, this.prefix, this.format, args);
    }
  }

  isLoggable(level: number) {
    hilog.isLoggable(this.domain, this.prefix, level);
  }
}

export default new Logger('HOK_HM');