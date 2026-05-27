"""Date/calendar utilities used throughout DSSAT.

DSSAT represents dates as a 7-digit integer ``YYYYDDD`` where *YYYY* is
the four-digit year and *DDD* is the 1-based day-of-year (Julian day).
"""

from __future__ import annotations

import datetime


def yrdoy_to_date(yrdoy: int) -> datetime.date:
    """Convert DSSAT YYYYDDD integer to a :class:`datetime.date`.

    Args:
        yrdoy: Date encoded as ``YYYYDDD``.

    Returns:
        Corresponding :class:`datetime.date`.

    Examples:
        >>> yrdoy_to_date(2020182)
        datetime.date(2020, 6, 30)
    """
    year, doy = divmod(yrdoy, 1000)
    return datetime.date(year, 1, 1) + datetime.timedelta(days=doy - 1)


def date_to_yrdoy(d: datetime.date) -> int:
    """Convert a :class:`datetime.date` to DSSAT YYYYDDD format.

    Args:
        d: Calendar date.

    Returns:
        Integer in YYYYDDD format.

    Examples:
        >>> date_to_yrdoy(datetime.date(2020, 6, 30))
        2020182
    """
    return d.year * 1000 + d.timetuple().tm_yday


def incdat(yrdoy: int, days: int) -> int:
    """Increment a DSSAT date by *days* calendar days.

    Args:
        yrdoy: Starting date, YYYYDDD.
        days: Number of days to add (may be negative).

    Returns:
        New date, YYYYDDD.
    """
    return date_to_yrdoy(yrdoy_to_date(yrdoy) + datetime.timedelta(days=days))


def timdif(yrdoy1: int, yrdoy2: int) -> int:
    """Number of calendar days between two DSSAT dates.

    Args:
        yrdoy1: Earlier date, YYYYDDD.
        yrdoy2: Later date, YYYYDDD.

    Returns:
        ``yrdoy2 - yrdoy1`` in calendar days (positive if yrdoy2 is later).
    """
    return (yrdoy_to_date(yrdoy2) - yrdoy_to_date(yrdoy1)).days


def yr_doy(yrdoy: int) -> tuple[int, int]:
    """Split a DSSAT YYYYDDD integer into (year, day-of-year).

    Args:
        yrdoy: Date, YYYYDDD.

    Returns:
        Tuple ``(year, doy)``.
    """
    year, doy = divmod(yrdoy, 1000)
    return year, doy


def doy_to_month_day(year: int, doy: int) -> tuple[int, int]:
    """Convert day-of-year to (month, day).

    Args:
        year: Four-digit year (needed for leap-year correctness).
        doy: 1-based day of year.

    Returns:
        Tuple ``(month, day)``.
    """
    d = datetime.date(year, 1, 1) + datetime.timedelta(days=doy - 1)
    return d.month, d.day
