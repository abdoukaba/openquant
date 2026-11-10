from __future__ import annotations
from enum import Enum
from typing import Callable
import pandas as pd


class Calendar(str, Enum):
    """Standard trading calendar definitions."""

    NYSE = "NYSE"
    LSE = "LSE"
    TSE = "TSE"
    EURONEXT = "EURONEXT"
    ALL_DAYS = "ALL"
    WEEKDAYS = "WEEKDAYS"


def _weekdays(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, end=end, freq="B")


def _all_days(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end)


def trading_days(
    start: str,
    end: str,
    calendar: Calendar | str = Calendar.NYSE,
) -> pd.DatetimeIndex:
    """
    Return trading days between start and end for the given calendar.
    Uses pandas_market_calendars if available, else falls back to weekdays.
    """
    cal = Calendar(calendar) if isinstance(calendar, str) else calendar
    try:
        import pandas_market_calendars as mcal  # optional

        cal_map = {
            Calendar.NYSE: "NYSE",
            Calendar.LSE: "LSE",
            Calendar.TSE: "JPX",
            Calendar.EURONEXT: "XPAR",
            Calendar.WEEKDAYS: None,
            Calendar.ALL_DAYS: None,
        }
        cal_name = cal_map.get(cal)
        if cal_name:
            exchange = mcal.get_calendar(cal_name)
            schedule = exchange.schedule(start_date=start, end_date=end)
            return pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()
    except ImportError:
        pass
    # Fallback
    if cal == Calendar.ALL_DAYS:
        return _all_days(start, end)
    return _weekdays(start, end)


class RebalanceSchedule:
    """
    Determines which dates trigger a portfolio rebalance.

    Supports:
        - frequency strings: "D", "W", "M", "Q", "A"
        - custom callable: fn(date) -> bool
        - explicit date list
    """

    def __init__(
        self,
        frequency: str | None = "M",
        dates: list[str] | pd.DatetimeIndex | None = None,
        custom_fn: Callable[[pd.Timestamp], bool] | None = None,
        day_of_week: int | None = None,  # 0=Mon..4=Fri
        day_of_month: int | None = None,  # 1-28
    ):
        self.frequency = frequency
        self._dates: pd.DatetimeIndex | None = (
            pd.DatetimeIndex(dates) if dates is not None else None
        )
        self.custom_fn = custom_fn
        self.day_of_week = day_of_week
        self.day_of_month = day_of_month
        if day_of_week is not None and day_of_week not in range(7):
            raise ValueError("day_of_week must be between 0 and 6")
        if day_of_month is not None and day_of_month not in range(1, 29):
            raise ValueError("day_of_month must be between 1 and 28")
        aliases = {"ME": "M", "QE": "Q", "YE": "A", "Y": "A"}
        self.frequency = aliases.get((frequency or "M").upper(), (frequency or "M").upper())
        if self.frequency.startswith("W-"):
            names = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
            try:
                self.day_of_week = names.index(self.frequency[2:])
            except ValueError as exc:
                raise ValueError("invalid weekly frequency") from exc
            self.frequency = "W"
        if self.frequency not in {"D", "B", "W", "M", "Q", "A"}:
            raise ValueError(f"unsupported rebalance frequency: {frequency}")

    @staticmethod
    def monthly() -> "RebalanceSchedule":
        return RebalanceSchedule(frequency="M")

    @staticmethod
    def quarterly() -> "RebalanceSchedule":
        return RebalanceSchedule(frequency="Q")

    @staticmethod
    def weekly(day: int = 4) -> "RebalanceSchedule":
        return RebalanceSchedule(frequency="W", day_of_week=day)

    @staticmethod
    def daily() -> "RebalanceSchedule":
        return RebalanceSchedule(frequency="D")

    @staticmethod
    def on_dates(dates) -> "RebalanceSchedule":
        return RebalanceSchedule(dates=dates)

    def is_rebalance_date(self, dt: pd.Timestamp, all_dates: pd.DatetimeIndex) -> bool:
        dt = pd.Timestamp(dt).tz_localize(None).normalize()
        all_dates = pd.DatetimeIndex(all_dates).tz_localize(None).normalize()
        if self._dates is not None:
            return dt in self._dates.tz_localize(None).normalize()
        if self.custom_fn is not None:
            return bool(self.custom_fn(dt))
        freq = (self.frequency or "M").upper()
        if freq in {"D", "B"}:
            return True
        if freq == "W":
            dow = self.day_of_week if self.day_of_week is not None else 4  # Friday
            next_days = all_dates[all_dates > dt]
            target = dt + pd.Timedelta(days=dow - dt.weekday())
            # If the requested weekday is a holiday, use the last session
            # preceding it. Never add a second Friday rebalance for weekly(day=0).
            return dt == target or (dt < target and len(next_days) > 0 and next_days[0] > target)
        if freq == "M":
            if self.day_of_month is not None:
                return dt.day == self.day_of_month
            # last trading day of month
            next_days = all_dates[all_dates > dt]
            return (
                pd.offsets.BMonthEnd().is_on_offset(dt)
                if len(next_days) == 0
                else next_days[0].to_period("M") != dt.to_period("M")
            )
        if freq == "Q":
            next_days = all_dates[all_dates > dt]
            return (
                pd.offsets.BQuarterEnd(startingMonth=12).is_on_offset(dt)
                if len(next_days) == 0
                else next_days[0].to_period("Q") != dt.to_period("Q")
            )
        if freq == "A":
            next_days = all_dates[all_dates > dt]
            return (
                pd.offsets.BYearEnd().is_on_offset(dt)
                if len(next_days) == 0
                else next_days[0].year != dt.year
            )
        return False

    def filter_dates(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Return the subset of dates that are rebalance dates."""
        return pd.DatetimeIndex([d for d in dates if self.is_rebalance_date(d, dates)])
