from datetime import tzinfo, timedelta

ZERO = timedelta(0)
HOUR = timedelta(hours=1)

class UTC(tzinfo):

    def utcoffset(self, dt):
        return ZERO

    def tzname(self, dt):
        return "UTC"

    def dst(self, dt):
        return ZERO
