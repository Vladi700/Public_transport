

def animate_queue_model(H, pos, model, hotspot, line_ids, line_to_color, frames=1000, interval_ms = 50, node_size=5):
    fig, ax = plt.subplots(figsize=(14, 14))

    for line_id in line_ids:
        edges_of_line = [(u,  v) for u, v, d in H.edges(data=True) if d.get("line_id") == line_id]
        nx.draw_networkx_edges(
            H, pos,
            edgelist=edges_of_line,
            width=2,
            edge_color=[line_to_color[line_id]],
            ax=ax
        )

        nodes = list(H.nodes)
        xy = np.array([pos[n] for n in nodes])
        def qmass(u):
            return sum(c.mass for c in model.waiting[u])
        q0 = np.array([qmass(n) for n in nodes],  dtype=float)
        sc = ax.scatter(
            xy[:, 0], xy[:, 1],
            s=node_size,
            c=q0,
            cmap='inferno',
            alpha=0.9
        )

        if hotspot in nodes:
            hx, hy = pos[hotspot_name]
            ax.scatter([hx], [hy], s=120, c='red', zorder=5)

        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Queue animation (node size/color = queued people)")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("Queue mass (people)")
    time_text = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top")
    def update(_):
        model.step()  # advance one tick in YOUR QueueModel

        q = np.array([qmass(n) for n in nodes], dtype=float)
        sc.set_array(q)

        # if you have model.tick or model.steps:
        tick = getattr(model, "tick", getattr(model, "steps", ""))
        time_text.set_text(f"tick={tick}")

        return (sc, time_text)

    anim = FuncAnimation(fig, update, frames=frames, interval=interval_ms, blit=False)
    plt.show()
    return anim