from pathlib import Path
import time

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler

MODEL_PATH = Path(__file__).resolve().parent
model, tokenizer = load(str(MODEL_PATH))

text = """The old lighthouse stood at the edge of the cliff, its white paint peeling after decades of salt wind and winter storms. Every evening, the keeper climbed the narrow spiral staircase, counting the one hundred and twelve steps out of habit, and lit the great lamp that swept its beam across the dark water. Fishing boats returning late relied on that light to find the safe channel between the rocks, and more than one captain had sent a letter of thanks after a bad night at sea.

In the mornings, the keeper wrote in a leather logbook: the weather, the ships that passed, the birds that nested in the gallery railings. Over forty years, the logbook had grown into a shelf of volumes, a quiet record of a coastline that changed slowly and then all at once. When the new harbor was built on the far side of the bay, the fishing fleet moved with it, and the channel below the cliff grew quiet. Some weeks, the keeper counted only two or three boats, and on calm nights the beam swept across water with nothing on it at all.

The town council debated closing the lighthouse. Automated beacons were cheaper, they said, and satellite navigation had made the old lights little more than decoration. A young engineer from the coast guard visited to inspect the lamp and the clockwork that turned it. She spent three days measuring, photographing, and asking questions. On the last evening, she climbed the stairs with the keeper and watched him light the lamp, and she asked why he still wrote everything down when no one read the logbooks anymore.

The keeper thought about it for a long time. He told her that the sea kept no records of its own. The storms came and went, the sandbars shifted, the birds arrived and left with the seasons, and none of it would be remembered unless someone wrote it down. He said the logbooks were not really about the lighthouse. They were about paying attention to a place long enough to notice how it changed.

The engineer's report recommended keeping the light. Her reasons were practical: the rocks below the cliff had not moved, and small boats without modern equipment still used the channel, especially in summer. But she added a final page that had nothing to do with engineering. She suggested that the logbooks be copied and stored at the town library, so that the record of forty years would not be lost if anything happened to the tower.

The council agreed. A volunteer from the library spent the next winter scanning every page, and in the spring the town held a small exhibition. Children traced the routes of old storms on printed maps. Retired fishermen found the nights they had been caught in bad weather, recorded in the keeper's careful handwriting, and told stories about them to anyone who would listen. A local teacher used the bird records to show her class how the timing of the spring migration had shifted over the decades.

The keeper attended the exhibition on its last day. He walked slowly past the displays, reading pages he had written years earlier, some of which he had forgotten entirely. When a reporter asked him how it felt to see his work on display, he said that it was strange to discover that keeping a record had also been a way of keeping company. Then he went home, because the evening was coming on, and the lamp still needed to be lit."""


def warm_up(prompt, temperature=0.5, top_p=0.9, repetition_penalty=1.1):
    sampler = make_sampler(temp=temperature, top_p=top_p)
    logits_processors = make_logits_processors(repetition_penalty=repetition_penalty)
    for _ in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=1,
        sampler=sampler,
        logits_processors=logits_processors,
    ):
        pass


def run(prompt, length, temperature=0.5, top_p=0.9, repetition_penalty=1.1):
    run_times = []
    sampler = make_sampler(temp=temperature, top_p=top_p)
    logits_processors = make_logits_processors(repetition_penalty=repetition_penalty)

    iterator = stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=length,
        sampler=sampler,
        logits_processors=logits_processors,
    )

    start = time.perf_counter()
    for idx, _ in enumerate(iterator):
        run_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        if idx + 1 >= length:
            break

    return run_times


if __name__ == "__main__":
    warm_up(text)
    times = run(text, 100)

    print("ttft:", times[0])
    print("itls:", times[1:])
    print("average itl:", sum(times[1:]) / len(times[1:]))
    print("average token per second:", 1 / (sum(times[1:]) / len(times[1:])))
