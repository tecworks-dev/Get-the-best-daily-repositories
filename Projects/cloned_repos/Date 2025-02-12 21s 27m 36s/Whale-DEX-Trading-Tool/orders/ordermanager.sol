// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Interface for fetching data from an external oracle (e.g., Chainlink)
interface IOracle {
    function getLatestPrice() external view returns (uint256);
}

contract OrderManager {
    address public owner;
    uint256 public orderAmount;    // Total order amount (USD)
    uint256 public splits;         // Number of parts to split the order
    uint256 public maxSlippage;    // Maximum slippage in basis points (e.g., 100 = 1%)
    uint256 public stopLossPrice;  // Stop-loss level
    uint256 public takeProfitPrice;// Take-profit level

    IOracle public priceOracle;

    event OrderExecuted(address indexed trader, uint256 orderPart, uint256 executedAmount);
    event ConfigUpdated(uint256 orderAmount, uint256 splits, uint256 maxSlippage);

    modifier onlyOwner() {
        require(msg.sender == owner, "Access restricted to the owner");
        _;
    }

    // Constructor sets initial parameters and connects the oracle
    constructor(
        uint256 _orderAmount,
        uint256 _splits,
        uint256 _maxSlippage,
        address _oracleAddress
    ) {
        owner = msg.sender;
        orderAmount = _orderAmount;
        splits = _splits;
        maxSlippage = _maxSlippage;
        priceOracle = IOracle(_oracleAddress);
    }

    // Function to update order configuration (owner-only)
    function updateConfig(
        uint256 _orderAmount,
        uint256 _splits,
        uint256 _maxSlippage
    ) external onlyOwner {
        orderAmount = _orderAmount;
        splits = _splits;
        maxSlippage = _maxSlippage;
        emit ConfigUpdated(orderAmount, splits, maxSlippage);
    }

    // Set risk management parameters: stop-loss and take-profit
    function setRiskParameters(uint256 _stopLoss, uint256 _takeProfit) external onlyOwner {
        stopLossPrice = _stopLoss;
        takeProfitPrice = _takeProfit;
    }

    // Main function to execute the order with splitting and price condition checks
    function executeOrder() external returns (bool) {
        uint256 currentPrice = priceOracle.getLatestPrice();
        require(
            currentPrice >= stopLossPrice && currentPrice <= takeProfitPrice,
            "Current price is outside the defined risk parameters"
        );
        uint256 splitOrderAmount = orderAmount / splits;
        // Simulate execution of each order part, considering a fee calculated as a portion of maxSlippage
        for (uint256 i = 0; i < splits; i++) {
            uint256 fee = (splitOrderAmount * maxSlippage) / 10000;
            uint256 executedAmount = splitOrderAmount - fee;
            emit OrderExecuted(msg.sender, splitOrderAmount, executedAmount);
        }
        return true;
    }
}
