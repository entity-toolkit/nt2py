from typing import Any, List, Tuple

from nt2.plotters.export import makeFrames


class _FakeFuture:
    def __init__(self, value: bool):
        self._value = value

    def result(self) -> bool:
        return self._value


class _FakeExecutor:
    def __init__(self):
        self.calls: List[Tuple[int, float, str, Any, Any]] = []

    def submit(self, func, ti, t, fpath, plot, data):
        self.calls.append((ti, t, fpath, plot, data))
        return _FakeFuture(func(ti, t, fpath, plot, data))


def test_make_frames_uses_executor_with_data(tmp_path, monkeypatch):
    ex = _FakeExecutor()

    monkeypatch.setattr(
        "loky.get_reusable_executor",
        lambda max_workers=None: ex,
    )

    called: List[float] = []

    def plot_frame(t, d):
        called.append(t)

    times = [0.0, 1.0, 2.0]
    result = makeFrames(
        plot=plot_frame, times=times, fpath=str(tmp_path), data={"ok": 1}
    )

    assert result == [True, True, True]
    assert len(ex.calls) == len(times)
    assert called == times
    for i in range(len(times)):
        assert (tmp_path / f"{i:05d}.png").exists()
