#include <stddef.h>
#include <stdint.h>

static constexpr uint8_t const pose_landmarker_task[] = {
#include "bin2h/_internal_pose_landmarker_full.task.inl"
};

extern uint8_t const *const pose_landmarker_task_base = pose_landmarker_task;

extern size_t const pose_landmarker_task_size = sizeof(pose_landmarker_task);
