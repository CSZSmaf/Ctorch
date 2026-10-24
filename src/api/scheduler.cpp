#include "api/common.h"

#include <algorithm>

using namespace dl::capi;

namespace {

dl_status create(dl_optimizer* optimizer, dl::optim::Schedule schedule, dl_scheduler** output) {
    *output = nullptr;
    return invoke([&] {
        auto value = std::make_unique<dl::optim::LRScheduler>(*optimizer->value, std::move(schedule));
        *output = new dl_scheduler{optimizer->value, std::move(value)};
    });
}

} // namespace

extern "C" {

dl_status dl_step_lr_create(dl_optimizer* optimizer, int64_t interval, double gamma, dl_scheduler** output) {
    return create(optimizer, dl::optim::StepSchedule{interval, gamma}, output);
}
dl_status dl_multistep_lr_create(dl_optimizer* optimizer, dl_shape milestones, double gamma, dl_scheduler** output) {
    *output = nullptr;
    return invoke([&] {
        auto value = std::make_unique<dl::optim::LRScheduler>(*optimizer->value,
            dl::optim::MultiStepSchedule{convert(milestones), gamma});
        *output = new dl_scheduler{optimizer->value, std::move(value)};
    });
}
dl_status dl_exponential_lr_create(dl_optimizer* optimizer, double gamma, dl_scheduler** output) {
    return create(optimizer, dl::optim::ExponentialSchedule{gamma}, output);
}
dl_status dl_cosine_lr_create(dl_optimizer* optimizer, int64_t period, double minimum_rate, dl_scheduler** output) {
    return create(optimizer, dl::optim::CosineSchedule{period, minimum_rate}, output);
}
dl_status dl_linear_lr_create(dl_optimizer* optimizer, int64_t steps, double start_factor, double end_factor, dl_scheduler** output) {
    return create(optimizer, dl::optim::LinearSchedule{steps, start_factor, end_factor}, output);
}
void dl_scheduler_release(dl_scheduler* scheduler) { delete scheduler; }
dl_status dl_scheduler_step(dl_scheduler* scheduler) { return invoke([&] { scheduler->value->step(); }); }
dl_status dl_scheduler_set_step(dl_scheduler* scheduler, int64_t steps) { return invoke([&] { scheduler->value->step(steps); }); }
int64_t dl_scheduler_get_step(const dl_scheduler* scheduler) { return scheduler->value->step_count(); }

dl_status dl_checkpoint_set_scheduler(dl_checkpoint* checkpoint, const char* name, const dl_scheduler* scheduler) {
    return invoke([&] {
        auto& entries = checkpoint->value.schedulers;
        auto found = std::find_if(entries.begin(), entries.end(), [&](const auto& entry) { return entry.first == name; });
        if (found == entries.end()) entries.emplace_back(name, scheduler->value->state_dict());
        else found->second = scheduler->value->state_dict();
    });
}
dl_status dl_checkpoint_restore_scheduler(const dl_checkpoint* checkpoint, const char* name, dl_scheduler* scheduler) {
    return invoke([&] {
        const auto& entries = checkpoint->value.schedulers;
        const auto found = std::find_if(entries.begin(), entries.end(), [&](const auto& entry) { return entry.first == name; });
        scheduler->value->load_state_dict(found->second);
    });
}

} // extern "C"
