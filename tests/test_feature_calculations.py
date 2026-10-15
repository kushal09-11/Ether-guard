"""
Automated unit tests for reference feature calculation engine.
Validates all 24 Canonical Common Features against manually verified numerical examples,
covering zero/one transactions, failed transactions, contract deployments,
Wei-to-ETH conversions, ERC-20 edge cases, and net flow vs balance.
"""

import pytest
from src.feature_calculator import calculate_common_features_from_etherscan, WEI_IN_ETHER


def test_feature_calculation_standard_wallet_manual_values():
    """
    Manually verified scenario:
    Wallet: 0xTarget
    Normal transactions:
      - Tx1 (ts=1000, from=0xSenderA, to=0xTarget, val=2.0 ETH, isError=0)
      - Tx2 (ts=1600, from=0xSenderB, to=0xTarget, val=6.0 ETH, isError=0)
      - Tx3 (ts=2200, from=0xTarget, to=0xRecipA, val=1.5 ETH, isError=0)
      - Tx4 (ts=3400, from=0xTarget, to=0xRecipB, val=3.5 ETH, isError=0)
      - Tx5 (ts=4000, from=0xTarget, to=0xRecipA, val=1.0 ETH, isError=0)
      - Tx6 (ts=4500, from=0xTarget, to="", contractAddress=0xContractA, val=0.0 ETH, isError=0)  # Contract deploy
      - Tx7 (ts=5000, from=0xTarget, to=0xRecipC, val=10.0 ETH, isError=1)  # Failed tx
    
    Expected manual calculations:
      - sent_tnx: 3 (Tx3, Tx4, Tx5)
      - received_tnx: 2 (Tx1, Tx2)
      - created_contracts_count: 1 (Tx6)
      - total_transactions_count: 3 + 2 + 1 = 6 (excludes failed Tx7)
      - time_diff_first_last_mins: (4500 - 1000) / 60 = 3500 / 60 = 58.33 mins
      - avg_min_between_sent_tnx: sent ts=[2200, 3400, 4000]. Span=1800s. (1800) / (60 * (3-1)) = 15.0 mins
      - avg_min_between_received_tnx: rec ts=[1000, 1600]. Span=600s. 600 / (60 * (2-1)) = 10.0 mins
      - unique_received_from_addresses: 2 (SenderA, SenderB)
      - unique_sent_to_addresses: 2 (RecipA, RecipB)
      - min_value_received_eth: 2.0
      - max_value_received_eth: 6.0
      - avg_value_received_eth: (2.0 + 6.0) / 2 = 4.0
      - total_ether_received: 8.0
      - min_value_sent_eth: 1.0
      - max_value_sent_eth: 3.5
      - avg_value_sent_eth: (1.5 + 3.5 + 1.0) / 3 = 2.0
      - total_ether_sent: 6.0
      - total_ether_balance: 8.0 - 6.0 = 2.0
    
    ERC-20 transfers:
      - Event1 (from=0xSenderX, to=0xTarget, contract=0xUSDT, symbol="USDT")
      - Event2 (from=0xSenderY, to=0xTarget, contract=0xUSDT, symbol="USDT")
      - Event3 (from=0xTarget, to=0xRecipZ, contract=0xDAI, symbol="DAI")
    
    Expected manual ERC-20 values:
      - erc20_total_tnx: 3
      - erc20_unique_sent_addr: 1 (RecipZ)
      - erc20_unique_received_addr: 2 (SenderX, SenderY)
      - erc20_unique_received_contract_addr: 1 (0xUSDT)
      - erc20_unique_sent_token_name: 1 (DAI)
      - erc20_unique_received_token_name: 1 (USDT)
    """
    target = "0xTarget"
    normal_txs = [
        {"timeStamp": 1000, "from": "0xSenderA", "to": target, "value": str(int(2.0 * WEI_IN_ETHER)), "isError": "0"},
        {"timeStamp": 1600, "from": "0xSenderB", "to": target, "value": str(int(6.0 * WEI_IN_ETHER)), "isError": "0"},
        {"timeStamp": 2200, "from": target, "to": "0xRecipA", "value": str(int(1.5 * WEI_IN_ETHER)), "isError": "0"},
        {"timeStamp": 3400, "from": target, "to": "0xRecipB", "value": str(int(3.5 * WEI_IN_ETHER)), "isError": "0"},
        {"timeStamp": 4000, "from": target, "to": "0xRecipA", "value": str(int(1.0 * WEI_IN_ETHER)), "isError": "0"},
        {"timeStamp": 4500, "from": target, "to": "", "contractAddress": "0xContractA", "value": "0", "isError": "0"},
        {"timeStamp": 5000, "from": target, "to": "0xRecipC", "value": str(int(10.0 * WEI_IN_ETHER)), "isError": "1"},  # Failed
    ]
    token_txs = [
        {"from": "0xSenderX", "to": target, "contractAddress": "0xUSDT", "tokenSymbol": "USDT"},
        {"from": "0xSenderY", "to": target, "contractAddress": "0xUSDT", "tokenSymbol": "USDT"},
        {"from": target, "to": "0xRecipZ", "contractAddress": "0xDAI", "tokenSymbol": "DAI"},
    ]

    features = calculate_common_features_from_etherscan(target, normal_txs, token_txs)

    assert features["sent_tnx"] == 3
    assert features["received_tnx"] == 2
    assert features["created_contracts_count"] == 1
    assert features["total_transactions_count"] == 6
    assert features["time_diff_first_last_mins"] == round(3500 / 60.0, 2)
    assert features["avg_min_between_sent_tnx"] == 15.0
    assert features["avg_min_between_received_tnx"] == 10.0
    assert features["unique_received_from_addresses"] == 2
    assert features["unique_sent_to_addresses"] == 2
    assert features["min_value_received_eth"] == 2.0
    assert features["max_value_received_eth"] == 6.0
    assert features["avg_value_received_eth"] == 4.0
    assert features["total_ether_received"] == 8.0
    assert features["min_value_sent_eth"] == 1.0
    assert features["max_value_sent_eth"] == 3.5
    assert features["avg_value_sent_eth"] == 2.0
    assert features["total_ether_sent"] == 6.0
    assert features["total_ether_balance"] == 2.0

    assert features["erc20_total_tnx"] == 3
    assert features["erc20_unique_sent_addr"] == 1
    assert features["erc20_unique_received_addr"] == 2
    assert features["erc20_unique_received_contract_addr"] == 1
    assert features["erc20_unique_sent_token_name"] == 1
    assert features["erc20_unique_received_token_name"] == 1


def test_edge_case_zero_sent_transactions():
    """Wallet with zero outgoing transactions."""
    target = "0xZeroSent"
    normal_txs = [
        {"timeStamp": 100, "from": "0xAlice", "to": target, "value": str(int(5.0 * WEI_IN_ETHER)), "isError": "0"},
    ]
    features = calculate_common_features_from_etherscan(target, normal_txs, [])

    assert features["sent_tnx"] == 0
    assert features["avg_min_between_sent_tnx"] == 0.0
    assert features["min_value_sent_eth"] == 0.0
    assert features["max_value_sent_eth"] == 0.0
    assert features["avg_value_sent_eth"] == 0.0
    assert features["total_ether_sent"] == 0.0
    assert features["unique_sent_to_addresses"] == 0
    assert features["total_ether_balance"] == 5.0


def test_edge_case_exactly_one_sent_transaction():
    """Wallet with exactly one outgoing transaction (no intervals)."""
    target = "0xOneSent"
    normal_txs = [
        {"timeStamp": 100, "from": target, "to": "0xBob", "value": str(int(2.5 * WEI_IN_ETHER)), "isError": "0"},
    ]
    features = calculate_common_features_from_etherscan(target, normal_txs, [])

    assert features["sent_tnx"] == 1
    assert features["avg_min_between_sent_tnx"] == 0.0  # N=1 -> interval is 0.0
    assert features["min_value_sent_eth"] == 2.5
    assert features["max_value_sent_eth"] == 2.5
    assert features["avg_value_sent_eth"] == 2.5
    assert features["total_ether_sent"] == 2.5
    assert features["unique_sent_to_addresses"] == 1


def test_edge_case_zero_received_transactions():
    """Wallet with zero incoming transactions."""
    target = "0xZeroReceived"
    normal_txs = [
        {"timeStamp": 100, "from": target, "to": "0xBob", "value": str(int(1.0 * WEI_IN_ETHER)), "isError": "0"},
    ]
    features = calculate_common_features_from_etherscan(target, normal_txs, [])

    assert features["received_tnx"] == 0
    assert features["avg_min_between_received_tnx"] == 0.0
    assert features["min_value_received_eth"] == 0.0
    assert features["max_value_received_eth"] == 0.0
    assert features["avg_value_received_eth"] == 0.0
    assert features["total_ether_received"] == 0.0
    assert features["unique_received_from_addresses"] == 0
    assert features["total_ether_balance"] == -1.0  # Received (0) - Sent (1) = -1.0


def test_edge_case_exactly_one_received_transaction():
    """Wallet with exactly one incoming transaction (no intervals)."""
    target = "0xOneReceived"
    normal_txs = [
        {"timeStamp": 100, "from": "0xAlice", "to": target, "value": str(int(4.75 * WEI_IN_ETHER)), "isError": "0"},
    ]
    features = calculate_common_features_from_etherscan(target, normal_txs, [])

    assert features["received_tnx"] == 1
    assert features["avg_min_between_received_tnx"] == 0.0  # N=1 -> interval is 0.0
    assert features["min_value_received_eth"] == 4.75
    assert features["max_value_received_eth"] == 4.75
    assert features["avg_value_received_eth"] == 4.75
    assert features["total_ether_received"] == 4.75
    assert features["unique_received_from_addresses"] == 1


def test_edge_case_single_or_zero_tx_time_diff():
    """Time diff between first and last must be 0.0 when count <= 1."""
    target = "0xSingleTx"
    # Case 0 tx
    f0 = calculate_common_features_from_etherscan(target, [], [])
    assert f0["time_diff_first_last_mins"] == 0.0
    assert f0["total_transactions_count"] == 0

    # Case 1 tx
    f1 = calculate_common_features_from_etherscan(target, [
        {"timeStamp": 1000, "from": target, "to": "0xBob", "value": "0", "isError": "0"}
    ], [])
    assert f1["time_diff_first_last_mins"] == 0.0
    assert f1["total_transactions_count"] == 1


def test_edge_case_failed_transactions_filtered():
    """Failed transactions (isError == '1') must be strictly excluded from valid transfers."""
    target = "0xErrorFilter"
    normal_txs = [
        {"timeStamp": 100, "from": target, "to": "0xBob", "value": str(int(10.0 * WEI_IN_ETHER)), "isError": "1"},  # Failed
        {"timeStamp": 200, "from": target, "to": "0xBob", "value": str(int(2.0 * WEI_IN_ETHER)), "isError": "0"},   # Success
    ]
    features = calculate_common_features_from_etherscan(target, normal_txs, [])

    assert features["sent_tnx"] == 1
    assert features["total_transactions_count"] == 1
    assert features["total_ether_sent"] == 2.0


def test_edge_case_contract_creation_partitioning():
    """Contract creations must increment created_contracts_count, not sent_tnx."""
    target = "0xContractDeployer"
    normal_txs = [
        {"timeStamp": 100, "from": target, "to": "", "contractAddress": "0xNewContract1", "value": "0", "isError": "0"},
        {"timeStamp": 200, "from": target, "to": "0x0000000000000000000000000000000000000000", "contractAddress": "0xNewContract2", "value": "0", "isError": "0"},
        {"timeStamp": 300, "from": target, "to": "0xBob", "value": str(int(1.0 * WEI_IN_ETHER)), "isError": "0"},
    ]
    features = calculate_common_features_from_etherscan(target, normal_txs, [])

    assert features["created_contracts_count"] == 2
    assert features["sent_tnx"] == 1
    assert features["total_transactions_count"] == 3
    assert features["unique_sent_to_addresses"] == 1


def test_edge_case_zero_erc20_transfers():
    """Zero ERC-20 transfers must produce 0 for all token features (no NaNs)."""
    target = "0xNoTokens"
    features = calculate_common_features_from_etherscan(target, [], token_transactions=None)

    assert features["erc20_total_tnx"] == 0
    assert features["erc20_unique_sent_addr"] == 0
    assert features["erc20_unique_received_addr"] == 0
    assert features["erc20_unique_received_contract_addr"] == 0
    assert features["erc20_unique_sent_token_name"] == 0
    assert features["erc20_unique_received_token_name"] == 0


def test_edge_case_net_flow_vs_actual_balance():
    """Net flow can be negative when sent > received, matching Kaggle dataset formula."""
    target = "0xDeficitWallet"
    normal_txs = [
        {"timeStamp": 100, "from": "0xAlice", "to": target, "value": str(int(3.0 * WEI_IN_ETHER)), "isError": "0"},
        {"timeStamp": 200, "from": target, "to": "0xBob", "value": str(int(10.0 * WEI_IN_ETHER)), "isError": "0"},
    ]
    features = calculate_common_features_from_etherscan(target, normal_txs, [])

    assert features["total_ether_received"] == 3.0
    assert features["total_ether_sent"] == 10.0
    assert features["total_ether_balance"] == -7.0  # Replicates received - sent
