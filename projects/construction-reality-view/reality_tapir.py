"""Run causal TAPIR on a query NPZ exported by reality_benchmark.py."""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from tapnet.models import tapir_model
from tapnet.utils import model_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("checkpoint", help="Official causal TAPIR checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()
    with np.load(args.input) as data:
        frames, queries = data["frames"], data["queries"]
    checkpoint = np.load(args.checkpoint, allow_pickle=True).item()
    model = tapir_model.ParameterizedTAPIR(
        params=checkpoint["params"], state=checkpoint["state"],
        tapir_kwargs={"use_causal_conv": True, "bilinear_interp_with_depthwise_conv": False},
    )

    @jax.jit
    def initialize(frame, query):
        grids = model.get_feature_grids(frame, is_training=False)
        return model.get_query_features(frame, is_training=False, query_points=query, feature_grids=grids)

    @jax.jit
    def predict(frame, features, state):
        grids = model.get_feature_grids(frame, is_training=False)
        result = model.estimate_trajectories(
            frame.shape[-3:-1], is_training=False, feature_grids=grids,
            query_features=features, query_points_in_video=None, query_chunk_size=64,
            causal_context=state, get_causal_context=True,
        )
        visible = model_utils.postprocess_occlusions(result["occlusion"][-1], result["expected_dist"][-1])
        return result["tracks"][-1], visible, result["causal_context"]

    height, width = frames.shape[1:3]
    scale = np.array([width, height]) / 256
    video = [model_utils.preprocess_frames(np.asarray(
        Image.fromarray(frame).resize((256, 256), Image.Resampling.BILINEAR)
    )[None, None]) for frame in frames]
    points = np.full((len(queries), len(frames), 2), np.nan)
    visible = np.zeros(points.shape[:2], dtype=bool)
    started = time.perf_counter()
    for first in np.unique(queries[:, 0]):
        indexes = np.flatnonzero(queries[:, 0] == first)
        xy = queries[indexes, 1:] / scale
        query = jnp.asarray(np.column_stack((np.zeros(len(indexes)), xy[:, ::-1]))[None], dtype=jnp.float32)
        features = initialize(video[first], query)
        state = model.construct_initial_causal_state(len(indexes), len(features.resolutions)-1)
        for index in range(first, len(frames)):
            xy, observed, state = predict(video[index], features, state)
            points[indexes, index] = np.asarray(xy)[0, :, 0] * scale
            visible[indexes, index] = np.asarray(observed)[0, :, 0]
    np.savez_compressed(args.output, points=points, visible=visible)
    print(f"{time.perf_counter() - started:.2f}s including compilation; backend={jax.default_backend()}")


if __name__ == "__main__":
    main()
