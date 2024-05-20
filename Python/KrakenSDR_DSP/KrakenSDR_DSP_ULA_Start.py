from KrakenSDR_DSP_ULA import krakensdr_dsp

processor = krakensdr_dsp(ipAddr="192.168.1.6", freq=1000.0, doaAlgortihm="MUSIC",
                      preprocessing="None", gain=[40.2,40.2,40.2,40.2,40.2], numSignals=1)

try:
    while True:
        processor.process_iq_samples()
except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    processor.stop()






