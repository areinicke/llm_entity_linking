from datetime import datetime

class PausableTimer():
    """
    timer.start() - should start the timer
    timer.pause() - should pause the timer
    timer.resume() - should resume the timer
    timer.get() - should return the current time
    """

    def __init__(self):
        self.timestarted = None
        self.timepaused = None
        self.paused = False

    def start(self):
        """ Starts an internal timer by recording the current time """
        self.timestarted = datetime.now()

    def pause(self):
        """ Pauses the timer """
        if self.timestarted is None:
            raise ValueError("Timer not started")
        if self.paused:
            return
        self.timepaused = datetime.now()
        self.paused = True

    def resume(self):
        """ Resumes the timer by adding the pause time to the start time """
        if self.timestarted is None:
            self.start()
            return
        if not self.paused:
            return
        pausetime = datetime.now() - self.timepaused
        self.timestarted = self.timestarted + pausetime
        self.paused = False

    def reset(self):
        """ Resets the timer to its initial state """
        self.timestarted = None
        self.timepaused = None
        self.paused = False

    def get(self):
        """ Returns a timedelta object showing the amount of time
            elapsed since the start time, less any pauses """
        if self.timestarted is None:
            timenow = datetime.now()
            return timenow - timenow  # Returns a zero timedelta if not started
        if self.paused:
            return self.timepaused - self.timestarted
        else:
            return datetime.now() - self.timestarted