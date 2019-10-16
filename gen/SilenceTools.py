import numpy as np

class SilenceTools:

    @staticmethod
    def cleanUp(audio, silence_level=0.02):
        y_max_inx = np.argmax(audio)
        temp_y = audio[:y_max_inx]
        temp_y = temp_y[::-1]
        y_start = None
        for i, a in enumerate(temp_y):
            if abs(temp_y[i]) <= silence_level:
                y_start = len(temp_y) - i
                break
        return audio[y_start:]
