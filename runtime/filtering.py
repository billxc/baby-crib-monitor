from collections import deque


class OccupancyFilter:
    """Causal 3-sample majority. Unknowns do not vote; gaps reset the state."""

    def __init__(self):
        self.window = deque(maxlen=3)
        self.state = 'unclear'

    def update(self, raw):
        if raw == 'missing':
            self.window.clear()
            self.state = 'unclear'
            return self.state
        self.window.append(raw)
        if self.window.count('yes') >= 2:
            self.state = 'yes'
        elif self.window.count('no') >= 2:
            self.state = 'no'
        elif self.state == 'unclear' and raw in ('yes', 'no'):
            self.state = raw
        return self.state
