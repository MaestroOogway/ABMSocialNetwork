# simulation.py
from typing import List, Dict, Tuple, Any, Optional
from agents import News
import csv, os


def simulate_multi_news(
    model,
    news_list: List[News],
    initial_seed_map: Dict[int, List[str]],
    max_iters: int = 20,
    save_aggregated_csv: str = None,
    save_detailed_log: str = None,
    visualizer: Optional[Any] = None
) -> Tuple[
    Dict[int, Dict[str, List[int]]],
    Dict[bool, Dict[str, List[int]]],
    int,
    List[Dict[str, Any]]
]:
    # Simulate multiple news items diffusing in parallel
    G = model.G

    # Map news_id -> News object
    news_by_id = {n.id: n for n in news_list}

    # Initialize frontiers and metrics
    frontiers: Dict[int, set] = {}
    metrics: Dict[int, Dict[str, List[int]]] = {}

    for news in news_list:
        nid = news.id
        seeds = [str(s) for s in initial_seed_map.get(nid, [])]
        frontiers[nid] = set(seeds)
        metrics[nid] = {"new_shared": [], "new_exposed": []}

        # Initialize seed state for each news
        for sid in seeds:
            agent = model.get_agent(sid)
            if nid not in agent.newsReceivedIds:
                agent.newsReceived.append(news)
                agent.newsReceivedIds.add(nid)
                agent.newsExposureCount[nid] = 1
            if nid not in agent.newsSharedIds:
                agent.newsShared.append(news)
                agent.newsSharedIds.add(nid)

    # Prepare aggregated time series
    aggregated: Dict[bool, Dict[str, List[int]]] = {
        True: {"shared": [0] * max_iters, "exposed": [0] * max_iters},
        False: {"shared": [0] * max_iters, "exposed": [0] * max_iters},
    }

    detailed_logs: List[Dict[str, Any]] = []

    actual_iters = 0

    for it in range(max_iters):
        any_active = False
        next_frontiers = {nid: set() for nid in frontiers}

        # Exposure tracking: news_id -> {receiver: set(sharers)}
        round_new_exposed: Dict[int, Dict[str, set]] = {nid: {} for nid in frontiers}

        # Collect exposure events
        for nid, sharers in frontiers.items():
            if not sharers:
                continue
            any_active = True
            news = news_by_id[nid]

            for sharer in sharers:
                for nb in G.neighbors(sharer):
                    nb = str(nb)
                    if nb == sharer:
                        continue

                    agent_nb = model.get_agent(nb)

                    # If already received, increase exposure count only
                    if nid in agent_nb.newsReceivedIds:
                        agent_nb.newsExposureCount[nid] = agent_nb.newsExposureCount.get(nid, 1) + 1
                        continue

                    # Register exposure for this iteration
                    round_new_exposed[nid].setdefault(nb, set()).add(sharer)

        # Process exposures for each news item
        for nid, exposed_map in round_new_exposed.items():
            news = news_by_id[nid]

            newly_exposed_count = sum(len(senders) for senders in exposed_map.values())
            newly_shared_count = 0

            for nb, sharer_set in exposed_map.items():
                agent = model.get_agent(nb)

                # Update agent state based on exposure
                agent.on_exposure(news, sender=None)

                # If agent shared, register for next frontier
                if nid in agent.newsSharedIds:
                    next_frontiers[nid].add(nb)
                    newly_shared_count += 1

                    # Detailed propagation logs
                    for sender in sharer_set:
                        detailed_logs.append({
                            "iteration": it,
                            "sender": sender,
                            "receiver": nb,
                            "news_id": nid,
                            "news_party": news.party,
                            "news_veracity": news.veracity
                        })

                    # Legacy propagation log for backward compatibility
                    model.news_propagation.append({
                        "sender_candidates": list(sharer_set),
                        "receiver_id": nb,
                        "news_id": nid,
                        "news_party": news.party,
                        "news_veracity": news.veracity,
                        "iteration": it
                    })

            # Update metrics
            metrics[nid]["new_exposed"].append(newly_exposed_count)
            metrics[nid]["new_shared"].append(newly_shared_count)

            # Update aggregated metrics
            ver = news.veracity
            aggregated[ver]["exposed"][it] += newly_exposed_count
            aggregated[ver]["shared"][it] += newly_shared_count

        # Update visualizer
        if visualizer is not None:
            visualizer.update(it, frontiers, aggregated, news_by_id)

        # Move to next iteration
        frontiers = next_frontiers
        actual_iters = it + 1

        # Stop if no new activity
        if not any_active:
            if visualizer is not None:
                visualizer.update(it, frontiers, aggregated, news_by_id)
            print(f"\nPropagation completed at iteration {it + 1} (no further activity).")
            break

    # Trim aggregated series to match actual iterations
    for ver in (True, False):
        aggregated[ver]["exposed"] = aggregated[ver]["exposed"][:actual_iters]
        aggregated[ver]["shared"] = aggregated[ver]["shared"][:actual_iters]

    # Save aggregated CSV output
    if save_aggregated_csv:
        os.makedirs(os.path.dirname(save_aggregated_csv), exist_ok=True)
        with open(save_aggregated_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "true_shared", "false_shared", "true_exposed", "false_exposed"])
            for i in range(actual_iters):
                writer.writerow([
                    i,
                    aggregated[True]["shared"][i],
                    aggregated[False]["shared"][i],
                    aggregated[True]["exposed"][i],
                    aggregated[False]["exposed"][i]
                ])

    # Save detailed propagation logs
    if save_detailed_log:
        os.makedirs(os.path.dirname(save_detailed_log), exist_ok=True)
        with open(save_detailed_log, "w", newline="") as f:
            fieldnames = ["iteration", "sender", "receiver", "news_id", "news_party", "news_veracity"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in detailed_logs:
                writer.writerow(row)

    return metrics, aggregated, actual_iters, detailed_logs
