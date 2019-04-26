from apscheduler.scheduler import Scheduler
from . import detection_files
import threading

check_fire = False
human_checker_not_present = False
human_checker_present = False
person_present_checker = False
object_detection = False
person = "Unknown"
time_constraint = 0
object_name = None


def set_fire_detection(p):
    global check_fire
    check_fire = p
    print("setting check_fire to " + str(check_fire))


def set_human_checker_not_present(p):
    global human_checker_not_present
    human_checker_not_present = p
    print("setting human_checker_not_present to " + str(human_checker_not_present))


def set_human_checker_present(p):
    global human_checker_present
    human_checker_present = p
    print("setting human_checker_present to " + str(human_checker_present))


def set_person_present_checker(p):
    global person_present_checker
    person_present_checker = p
    print("setting person_present_checker to " + str(person_present_checker))
    if p is True:
        detection_files.timer = 0


def set_object_detection(p):
    global object_detection
    object_detection = p
    print("setting object_detection to " + str(object_detection))


# def bleh(p):
#     print(p)

def run_scheduler(cff, fst, fet, hnpf, hnpst, hnpet, hpf, hpst, hpet, ppc, ppcst, ppcet, pn, tc, od, on, odst, odet):

    sched = Scheduler()
    sched.start()
    # sched.add_date_job(bleh, '2019-04-24 16:20:40', ['Blehhhh!!'])

    if ppc is True:
        global person
        person = pn
        global time_constraint
        time_constraint = tc

        if ppcst is not None:
            temp = ppcst.replace('T', ' ')
            temp = temp.replace('Z', ' ')
            sched.add_date_job(set_person_present_checker, temp, [True])
        if ppcet is not None:
            temp = ppcet.replace('T', ' ')
            temp = temp.replace('Z', ' ')
            sched.add_date_job(set_person_present_checker, temp, [False])
        else:
            pass

    # ==================================================================================================================

    t2 = threading.Thread(target=detection_files.all_operations)
    t2.start()

    # ==================================================================================================================

    if cff is True:
        if fst is not None:
            temp = fst.replace('T', ' ')
            temp = temp.replace('Z', ' ')
            sched.add_date_job(set_fire_detection, temp, [True])
        if fet is not None:
            temp = fet.replace('T', ' ')
            temp = temp.replace('Z', ' ')
            sched.add_date_job(set_fire_detection, temp, [False])
        else:
            pass

    if hnpf is True:
        if hnpst is not None:
            temp = hnpst.replace('T', ' ')
            temp = temp.replace('Z', ' ')
            sched.add_date_job(set_human_checker_not_present, temp, [True])
        if hnpet is not None:
            temp = hnpet.replace('T', ' ')
            temp = temp.replace('Z', ' ')
            sched.add_date_job(set_human_checker_not_present, temp, [False])
        else:
            pass

    if hpf is True:
        if hpst is not None:
            temp = hpst.replace('T', ' ')
            temp = temp.replace('Z', ' ')
            sched.add_date_job(set_human_checker_present, temp, [True])
        if hpet is not None:
            temp = hpet.replace('T', ' ')
            temp = temp.replace('Z', ' ')
            sched.add_date_job(set_human_checker_present, temp, [False])
        else:
            pass

    if od is True:
        if odst is not None:
            global object_name
            object_name = on
            print(object_name)
            temp = odst.replace('T', ' ')
            temp = temp.replace('Z', ' ')
            sched.add_date_job(set_object_detection, temp, [True])
        if odet is not None:
            temp = odet.replace('T', ' ')
            temp = temp.replace('Z', ' ')
            sched.add_date_job(set_object_detection, temp, [False])
        else:
            pass

    # if ppc is True:
    #     global person
    #     person = pn
    #     global time_constraint
    #     time_constraint = tc
    #     if ppcst is not None:
    #         temp = ppcst.replace('T', ' ')
    #         temp = temp.replace('Z', ' ')
    #         sched.add_date_job(set_person_present_checker, temp, [True])
    #     if ppcet is not None:
    #         temp = ppcet.replace('T', ' ')
    #         temp = temp.replace('Z', ' ')
    #         sched.add_date_job(set_person_present_checker, temp, [False])
    #     else:
    #         pass

