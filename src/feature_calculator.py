"""
Reference Feature Calculation Engine for Ethereum Wallet Fraud Detection.
Implements mathematically verified calculations for all 24 Canonical Common Features
directly from raw Etherscan API V2 transaction lists (`txlist` and `tokentx`).
Provides exact edge-case handling for zero/one transactions, failed transactions,
contract deployments, Wei-to-ETH conversions, and ERC-20 token statistics.
"""

from typing import List, Dict, Any, Optional


WEI_IN_ETHER = 10**18


def calculate_common_features_from_etherscan(
    wallet_address: str,
    normal_transactions: List[Dict[str, Any]],
    token_transactions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Computes the 24 Canonical Common Features from raw Etherscan transaction lists.
    
    Args:
        wallet_address: Target Ethereum address (0x...)
        normal_transactions: List of records from `module=account&action=txlist`
        token_transactions: List of records from `module=account&action=tokentx` (or None)
        
    Returns:
        Dict mapping each of the 24 canonical feature names to its computed value.
    """
    target = wallet_address.lower().strip()
    token_txs = token_transactions if token_transactions is not None else []

    # 1. Filter normal transactions: only successful transactions (isError == "0")
    successful_normal_txs = [
        tx for tx in normal_transactions
        if str(tx.get("isError", "0")).strip() == "0"
    ]

    # Partition normal transactions into:
    # - Contract creations: from == target and (to is None/empty/0x or contractAddress != "")
    # - Sent transactions: from == target and to is standard address (not contract creation)
    # - Received transactions: to == target
    contract_creations = []
    sent_txs = []
    received_txs = []

    for tx in successful_normal_txs:
        from_addr = str(tx.get("from", "")).lower().strip()
        to_addr = str(tx.get("to", "")).lower().strip()
        created_contract = str(tx.get("contractAddress", "")).strip()

        null_addresses = ["", "0x", "none", "0x0000000000000000000000000000000000000000"]
        is_contract_creation = (
            from_addr == target and (
                to_addr in null_addresses or bool(created_contract and to_addr in null_addresses)
            )
        )

        if is_contract_creation:
            contract_creations.append(tx)
        elif from_addr == target:
            sent_txs.append(tx)
        elif to_addr == target:
            received_txs.append(tx)

    # Helper: Convert Wei to Ether
    def wei_to_eth(val_any: Any) -> float:
        try:
            return float(val_any) / WEI_IN_ETHER
        except (ValueError, TypeError):
            return 0.0

    # Helper: Safe timestamp extraction
    def get_timestamp(tx: Dict[str, Any]) -> int:
        try:
            return int(tx.get("timeStamp", 0))
        except (ValueError, TypeError):
            return 0

    # --- Feature Calculations ---

    # 1. Sent transactions count
    sent_tnx = len(sent_txs)

    # 2. Received transactions count
    received_tnx = len(received_txs)

    # 3. Created contracts count
    created_contracts_count = len(contract_creations)

    # 4. Total normal transactions (matches Kaggle definition: sent + received + contract creations)
    total_transactions_count = sent_tnx + received_tnx + created_contracts_count

    # 5. Sent transaction time interval:
    # If sent_tnx <= 1, interval is 0.0. Otherwise (last - first) / (60 * (count - 1))
    if sent_tnx > 1:
        sent_sorted_ts = sorted([get_timestamp(tx) for tx in sent_txs])
        avg_min_between_sent_tnx = (sent_sorted_ts[-1] - sent_sorted_ts[0]) / (60.0 * (sent_tnx - 1))
    else:
        avg_min_between_sent_tnx = 0.0

    # 6. Received transaction time interval:
    # If received_tnx <= 1, interval is 0.0. Otherwise (last - first) / (60 * (count - 1))
    if received_tnx > 1:
        rec_sorted_ts = sorted([get_timestamp(tx) for tx in received_txs])
        avg_min_between_received_tnx = (rec_sorted_ts[-1] - rec_sorted_ts[0]) / (60.0 * (received_tnx - 1))
    else:
        avg_min_between_received_tnx = 0.0

    # 7. Total observation lifespan (Time Diff first and last in minutes)
    # Measured across all valid normal transactions
    all_normal_ts = [get_timestamp(tx) for tx in successful_normal_txs]
    if len(all_normal_ts) > 1:
        time_diff_first_last_mins = (max(all_normal_ts) - min(all_normal_ts)) / 60.0
    else:
        time_diff_first_last_mins = 0.0

    # 8. Unique Sent To Addresses
    unique_sent_to_addresses = len(set(
        str(tx.get("to", "")).lower().strip() for tx in sent_txs
        if tx.get("to")
    ))

    # 9. Unique Received From Addresses
    unique_received_from_addresses = len(set(
        str(tx.get("from", "")).lower().strip() for tx in received_txs
        if tx.get("from")
    ))

    # 10. Sent ETH values
    sent_eth_values = [wei_to_eth(tx.get("value", 0)) for tx in sent_txs]
    if sent_tnx > 0:
        min_value_sent_eth = min(sent_eth_values)
        max_value_sent_eth = max(sent_eth_values)
        avg_value_sent_eth = sum(sent_eth_values) / sent_tnx
        total_ether_sent = sum(sent_eth_values)
    else:
        min_value_sent_eth = 0.0
        max_value_sent_eth = 0.0
        avg_value_sent_eth = 0.0
        total_ether_sent = 0.0

    # 11. Received ETH values
    rec_eth_values = [wei_to_eth(tx.get("value", 0)) for tx in received_txs]
    if received_tnx > 0:
        min_value_received_eth = min(rec_eth_values)
        max_value_received_eth = max(rec_eth_values)
        avg_value_received_eth = sum(rec_eth_values) / received_tnx
        total_ether_received = sum(rec_eth_values)
    else:
        min_value_received_eth = 0.0
        max_value_received_eth = 0.0
        avg_value_received_eth = 0.0
        total_ether_received = 0.0

    # 12. Total Ether Balance (Formula: total_ether_received - total_ether_sent)
    # Matches the Kaggle dataset's synthetic net flow definition
    total_ether_balance = total_ether_received - total_ether_sent

    # --- ERC-20 Token Features (6 features from tokentx) ---
    erc20_total_tnx = len(token_txs)

    erc20_sent = [
        tx for tx in token_txs
        if str(tx.get("from", "")).lower().strip() == target
    ]
    erc20_received = [
        tx for tx in token_txs
        if str(tx.get("to", "")).lower().strip() == target
    ]

    # Unique sent recipient addresses
    erc20_unique_sent_addr = len(set(
        str(tx.get("to", "")).lower().strip() for tx in erc20_sent
        if tx.get("to")
    ))

    # Unique received sender addresses
    erc20_unique_received_addr = len(set(
        str(tx.get("from", "")).lower().strip() for tx in erc20_received
        if tx.get("from")
    ))

    # Unique received contract addresses (token contracts held/received)
    erc20_unique_received_contract_addr = len(set(
        str(tx.get("contractAddress", "")).lower().strip() for tx in erc20_received
        if tx.get("contractAddress")
    ))

    # Unique sent token symbols
    erc20_unique_sent_token_name = len(set(
        str(tx.get("tokenSymbol", "")).strip() for tx in erc20_sent
        if tx.get("tokenSymbol")
    ))

    # Unique received token symbols
    erc20_unique_received_token_name = len(set(
        str(tx.get("tokenSymbol", "")).strip() for tx in erc20_received
        if tx.get("tokenSymbol")
    ))

    return {
        "avg_min_between_sent_tnx": round(float(avg_min_between_sent_tnx), 2),
        "avg_min_between_received_tnx": round(float(avg_min_between_received_tnx), 2),
        "time_diff_first_last_mins": round(float(time_diff_first_last_mins), 2),
        "sent_tnx": int(sent_tnx),
        "received_tnx": int(received_tnx),
        "created_contracts_count": int(created_contracts_count),
        "unique_received_from_addresses": int(unique_received_from_addresses),
        "unique_sent_to_addresses": int(unique_sent_to_addresses),
        "min_value_received_eth": float(min_value_received_eth),
        "max_value_received_eth": float(max_value_received_eth),
        "avg_value_received_eth": float(avg_value_received_eth),
        "min_value_sent_eth": float(min_value_sent_eth),
        "max_value_sent_eth": float(max_value_sent_eth),
        "avg_value_sent_eth": float(avg_value_sent_eth),
        "total_transactions_count": int(total_transactions_count),
        "total_ether_sent": float(total_ether_sent),
        "total_ether_received": float(total_ether_received),
        "total_ether_balance": float(total_ether_balance),
        "erc20_total_tnx": int(erc20_total_tnx),
        "erc20_unique_sent_addr": int(erc20_unique_sent_addr),
        "erc20_unique_received_addr": int(erc20_unique_received_addr),
        "erc20_unique_received_contract_addr": int(erc20_unique_received_contract_addr),
        "erc20_unique_sent_token_name": int(erc20_unique_sent_token_name),
        "erc20_unique_received_token_name": int(erc20_unique_received_token_name),
    }
