# ------------------------------------------------------------
# 1. INSTALL REQUIRED LIBRARIES
# ------------------------------------------------------------

!pip -q install matplotlib numpy pandas opencv-python-headless

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import subprocess
from IPython.display import display, Video

print("Libraries installed successfully.")


# ============================================================
# 2. DOWNLOAD SPORTS VIDEO
# ============================================================

video_url = (
    "https://d1l0eyz2lfj4xa.cloudfront.net/"
    "playOnPoly/693d86be47fd95f56c93ef98/"
    "cloud_hls/0_hd_hls.m3u8"
)

input_video = "/content/sports_video.mp4"

print("\nDownloading 10-second sports video...")

# Download short clip
result = subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-i",
        video_url,
        "-t",
        "10",
        "-vf",
        "scale=640:-2",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        input_video
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

if not os.path.exists(input_video):
    print("Video download failed.")
    print(result.stderr.decode()[-1000:])
    raise Exception("Could not download video.")

print("Sports video downloaded successfully.")


# ============================================================
# 3. VIDEO INFORMATION
# ============================================================

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    raise Exception("Could not open video.")

original_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

duration = (
    total_frames / original_fps
    if original_fps > 0 else 0
)

print("\n========== VIDEO INFORMATION ==========")
print("Resolution :", width, "x", height)
print("FPS        :", round(original_fps, 2))
print("Frames     :", total_frames)
print("Duration   :", round(duration, 2), "seconds")
print("=======================================")

cap.release()


# ============================================================
# 4. FUNCTION 1: OPTICAL FLOW
# ============================================================

def optical_flow_tracking(video_path):

    cap = cv2.VideoCapture(video_path)

    ret, first_frame = cap.read()

    if not ret:
        return None

    gray_old = cv2.cvtColor(
        first_frame,
        cv2.COLOR_BGR2GRAY
    )

    # Good feature parameters
    feature_params = dict(
        maxCorners=80,
        qualityLevel=0.3,
        minDistance=10,
        blockSize=7
    )

    p0 = cv2.goodFeaturesToTrack(
        gray_old,
        mask=None,
        **feature_params
    )

    if p0 is None:
        cap.release()
        return None

    # Lucas-Kanade parameters
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(
            cv2.TERM_CRITERIA_EPS |
            cv2.TERM_CRITERIA_COUNT,
            10,
            0.03
        )
    )

    frame_count = 0
    total_points = 0
    successful_points = 0

    trajectory_x = []
    trajectory_y = []

    output_path = "/content/optical_flow_output.mp4"

    h, w = gray_old.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        original_fps,
        (w, h)
    )

    start_time = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        gray_new = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # Optical Flow
        p1, status, error = cv2.calcOpticalFlowPyrLK(
            gray_old,
            gray_new,
            p0,
            None,
            **lk_params
        )

        if p1 is not None:

            good_new = p1[status == 1]
            good_old = p0[status == 1]

            total_points += len(p0)
            successful_points += len(good_new)

            for new, old in zip(
                good_new,
                good_old
            ):

                x1, y1 = new.ravel()
                x0, y0 = old.ravel()

                x1 = int(x1)
                y1 = int(y1)
                x0 = int(x0)
                y0 = int(y0)

                # Draw motion vector
                cv2.arrowedLine(
                    frame,
                    (x0, y0),
                    (x1, y1),
                    (255, 0, 0),
                    1,
                    tipLength=0.3
                )

                # Draw tracking point
                cv2.circle(
                    frame,
                    (x1, y1),
                    3,
                    (0, 255, 0),
                    -1
                )

                trajectory_x.append(x1)
                trajectory_y.append(y1)

            if len(good_new) > 0:

                p0 = good_new.reshape(
                    -1, 1, 2
                )

            else:

                p0 = cv2.goodFeaturesToTrack(
                    gray_new,
                    mask=None,
                    **feature_params
                )

                if p0 is None:
                    break

        cv2.putText(
            frame,
            "OPTICAL FLOW",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        writer.write(frame)

        frame_count += 1

        gray_old = gray_new

    end_time = time.time()

    cap.release()
    writer.release()

    processing_time = end_time - start_time

    fps = (
        frame_count / processing_time
        if processing_time > 0 else 0
    )

    consistency = (
        successful_points /
        total_points * 100
        if total_points > 0 else 0
    )

    return {
        "method": "Optical Flow",
        "frames": frame_count,
        "time": processing_time,
        "fps": fps,
        "consistency": consistency,
        "x": trajectory_x,
        "y": trajectory_y,
        "output": output_path
    }


# ============================================================
# 5. FUNCTION 2: MOTION ESTIMATION
#    FAST BLOCK-BASED MOTION ESTIMATION
# ============================================================

def motion_estimation(video_path):

    cap = cv2.VideoCapture(video_path)

    ret, previous_frame = cap.read()

    if not ret:
        return None

    previous_gray = cv2.cvtColor(
        previous_frame,
        cv2.COLOR_BGR2GRAY
    )

    h, w = previous_gray.shape

    output_path = "/content/motion_estimation_output.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        original_fps,
        (w, h)
    )

    # Use a grid of points instead of expensive
    # pixel-by-pixel block matching.
    step = 20

    previous_points = []

    for y in range(
        step,
        h - step,
        step
    ):

        for x in range(
            step,
            w - step,
            step
        ):

            previous_points.append(
                [x, y]
            )

    previous_points = np.array(
        previous_points,
        dtype=np.float32
    ).reshape(-1, 1, 2)

    frame_count = 0

    total_points = 0
    successful_points = 0

    trajectory_x = []
    trajectory_y = []

    start_time = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        current_gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # Fast motion estimation using sparse
        # feature displacement.
        current_points, status, error = (
            cv2.calcOpticalFlowPyrLK(
                previous_gray,
                current_gray,
                previous_points,
                None,
                winSize=(10, 10),
                maxLevel=1,
                criteria=(
                    cv2.TERM_CRITERIA_EPS |
                    cv2.TERM_CRITERIA_COUNT,
                    5,
                    0.03
                )
            )
        )

        if current_points is not None:

            good_current = (
                current_points[status == 1]
            )

            good_previous = (
                previous_points[status == 1]
            )

            total_points += len(
                previous_points
            )

            successful_points += len(
                good_current
            )

            # Draw motion vectors
            for new, old in zip(
                good_current,
                good_previous
            ):

                x1, y1 = new.ravel()
                x0, y0 = old.ravel()

                x1 = int(x1)
                y1 = int(y1)
                x0 = int(x0)
                y0 = int(y0)

                # Only draw significant movement
                distance = np.sqrt(
                    (x1 - x0) ** 2 +
                    (y1 - y0) ** 2
                )

                if distance > 1:

                    cv2.arrowedLine(
                        frame,
                        (x0, y0),
                        (x1, y1),
                        (0, 0, 255),
                        1,
                        tipLength=0.3
                    )

                    cv2.circle(
                        frame,
                        (x1, y1),
                        2,
                        (0, 255, 0),
                        -1
                    )

                    trajectory_x.append(
                        x1
                    )

                    trajectory_y.append(
                        y1
                    )

            if len(good_current) > 0:

                previous_points = (
                    good_current
                    .reshape(-1, 1, 2)
                )

            else:

                previous_points = (
                    np.array(
                        [
                            [x, y]
                            for y in range(
                                step,
                                h - step,
                                step
                            )
                            for x in range(
                                step,
                                w - step,
                                step
                            )
                        ],
                        dtype=np.float32
                    ).reshape(-1, 1, 2)
                )

        cv2.putText(
            frame,
            "MOTION ESTIMATION",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        writer.write(frame)

        frame_count += 1

        previous_gray = current_gray

    end_time = time.time()

    cap.release()
    writer.release()

    processing_time = end_time - start_time

    fps = (
        frame_count / processing_time
        if processing_time > 0 else 0
    )

    consistency = (
        successful_points /
        total_points * 100
        if total_points > 0 else 0
    )

    return {
        "method": "Motion Estimation",
        "frames": frame_count,
        "time": processing_time,
        "fps": fps,
        "consistency": consistency,
        "x": trajectory_x,
        "y": trajectory_y,
        "output": output_path
    }


# ============================================================
# 6. RUN OPTICAL FLOW
# ============================================================

print("\n======================================")
print("RUNNING OPTICAL FLOW")
print("======================================")

optical = optical_flow_tracking(
    input_video
)

print("Optical Flow completed.")


# ============================================================
# 7. RUN MOTION ESTIMATION
# ============================================================

print("\n======================================")
print("RUNNING MOTION ESTIMATION")
print("======================================")

motion = motion_estimation(
    input_video
)

print("Motion Estimation completed.")


# ============================================================
# 8. CREATE RESULTS TABLE
# ============================================================

results = pd.DataFrame({

    "Method": [
        optical["method"],
        motion["method"]
    ],

    "Frames": [
        optical["frames"],
        motion["frames"]
    ],

    "Processing Time (seconds)": [
        optical["time"],
        motion["time"]
    ],

    "FPS": [
        optical["fps"],
        motion["fps"]
    ],

    "Tracking Consistency (%)": [
        optical["consistency"],
        motion["consistency"]
    ]
})

results[
    "Processing Time (seconds)"
] = results[
    "Processing Time (seconds)"
].round(2)

results["FPS"] = results[
    "FPS"
].round(2)

results[
    "Tracking Consistency (%)"
] = results[
    "Tracking Consistency (%)"
].round(2)


print("\n======================================")
print("FINAL EXPERIMENTAL RESULTS")
print("======================================")

display(results)


# ============================================================
# 9. SAVE RESULTS
# ============================================================

csv_path = "/content/motion_tracking_results.csv"

results.to_csv(
    csv_path,
    index=False
)

print(
    "\nResults saved to:",
    csv_path
)


# ============================================================
# 10. GRAPH 1 - TRACKING CONSISTENCY
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    results["Method"],
    results["Tracking Consistency (%)"]
)

plt.title(
    "Optical Flow vs Motion Estimation\n"
    "Tracking Consistency"
)

plt.xlabel(
    "Method"
)

plt.ylabel(
    "Tracking Consistency (%)"
)

plt.ylim(0, 100)

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.savefig(
    "/content/tracking_consistency.png",
    dpi=300
)

plt.show()


# ============================================================
# 11. GRAPH 2 - FPS
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    results["Method"],
    results["FPS"]
)

plt.title(
    "Optical Flow vs Motion Estimation\n"
    "Processing Speed"
)

plt.xlabel(
    "Method"
)

plt.ylabel(
    "Frames Per Second (FPS)"
)

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.savefig(
    "/content/fps_comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# 12. GRAPH 3 - PROCESSING TIME
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    results["Method"],
    results["Processing Time (seconds)"]
)

plt.title(
    "Optical Flow vs Motion Estimation\n"
    "Processing Time"
)

plt.xlabel(
    "Method"
)

plt.ylabel(
    "Processing Time (seconds)"
)

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.savefig(
    "/content/processing_time.png",
    dpi=300
)

plt.show()


# ============================================================
# 13. GRAPH 4 - ACCURACY vs SPEED
# ============================================================

# Normalize FPS to 0-100 so that FPS and consistency
# can be visualized on the same scale.

fps_values = results["FPS"].values

max_fps = max(
    fps_values
) if max(fps_values) > 0 else 1

normalized_fps = (
    fps_values /
    max_fps *
    100
)

accuracy_values = results[
    "Tracking Consistency (%)"
].values

x = np.arange(
    len(results["Method"])
)

bar_width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - bar_width / 2,
    accuracy_values,
    bar_width,
    label="Tracking Consistency"
)

plt.bar(
    x + bar_width / 2,
    normalized_fps,
    bar_width,
    label="Normalized FPS"
)

plt.xticks(
    x,
    results["Method"]
)

plt.ylabel(
    "Normalized Performance (%)"
)

plt.xlabel(
    "Motion Tracking Method"
)

plt.title(
    "Accuracy-Speed Trade-off"
)

plt.legend()

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.savefig(
    "/content/accuracy_speed_tradeoff.png",
    dpi=300
)

plt.show()


# ============================================================
# 14. GRAPH 5 - OPTICAL FLOW TRAJECTORY
# ============================================================

if len(optical["x"]) > 0:

    plt.figure(figsize=(10, 6))

    plt.plot(
        optical["x"],
        optical["y"],
        linewidth=1
    )

    plt.title(
        "Optical Flow Motion Trajectory"
    )

    plt.xlabel(
        "X Position"
    )

    plt.ylabel(
        "Y Position"
    )

    plt.gca().invert_yaxis()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "/content/optical_flow_trajectory.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 15. GRAPH 6 - MOTION ESTIMATION TRAJECTORY
# ============================================================

if len(motion["x"]) > 0:

    plt.figure(figsize=(10, 6))

    plt.plot(
        motion["x"],
        motion["y"],
        linewidth=1
    )

    plt.title(
        "Motion Estimation Trajectory"
    )

    plt.xlabel(
        "X Position"
    )

    plt.ylabel(
        "Y Position"
    )

    plt.gca().invert_yaxis()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "/content/motion_estimation_trajectory.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 16. FINAL AUTOMATIC ANALYSIS
# ============================================================

best_speed = results.loc[
    results["FPS"].idxmax(),
    "Method"
]

best_consistency = results.loc[
    results["Tracking Consistency (%)"].idxmax(),
    "Method"
]

print("\n======================================")
print("ENGINEERING ANALYSIS")
print("======================================")

print(
    "Higher processing speed :",
    best_speed
)

print(
    "Higher tracking consistency :",
    best_consistency
)

print("\nInterpretation:")

if best_speed == "Motion Estimation":

    print(
        "Motion Estimation achieved the higher "
        "processing speed in this experiment."
    )

else:

    print(
        "Optical Flow achieved the higher "
        "processing speed in this experiment."
    )

if best_consistency == "Optical Flow":

    print(
        "Optical Flow achieved the higher "
        "tracking consistency in this experiment."
    )

else:

    print(
        "Motion Estimation achieved the higher "
        "tracking consistency in this experiment."
    )

print(
    "\nThe final engineering decision should "
    "consider both tracking quality and "
    "real-time processing requirements."
)

print("\n======================================")
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("======================================")