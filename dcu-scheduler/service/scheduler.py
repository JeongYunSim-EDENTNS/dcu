import asyncio
import peewee_async
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from utils.logger import get_logger
from utils.console import bold, invert
from service.task import run_task
from models.tables import DCUEmbeddingTable, SchedulerConfigTable
from utils.database import manager_pgvector

# Logger 설정
logger = get_logger(name="scheduler", log_dir="logs")

# APScheduler 설정
scheduler = AsyncIOScheduler()  
stop_event = asyncio.Event()

async def check_new_data():
    """
    공지사항 테이블에서 새로운 데이터를 확인하고 처리합니다.
    """
    job_id = "check_new_data"
    logger.info(f"{invert(f' {job_id} ')}")

    try:
        logger.info(f"{invert(f' {job_id} ')} {bold('새로운 데이터 확인 및 처리를 시작합니다.')}")
        # watcher_active 상태 확인
        query = SchedulerConfigTable.select(SchedulerConfigTable.watcher_active).where(SchedulerConfigTable.watcher_active == True)
        result = await manager_pgvector.execute(query)
        
        if result:
            logger.info(f"{invert(f' {job_id} ')} {bold('watcher is active')}")
            await run_task()
            
        else:
            logger.info(f"{invert(f' {job_id} ')} {bold('watcher_active 상태가 False입니다. 이번 작업을 건너뜁니다.')}")
        
    except Exception as e:
        logger.error(f"{invert(f' {job_id} ')} {bold('신규 데이터 확인 및 처리 중 오류 발생')} / {e}")



async def init_scheduler_config():
    """
    스케줄러 설정을 초기화합니다.
    watcher_active 상태를 true로 설정합니다.
    """
    job_id = "init_scheduler_config"
    logger.info(f"{invert(f' {job_id} ')} {bold('스케줄러 설정을 초기화합니다.')}")
    query = SchedulerConfigTable.update(watcher_active=True).where(SchedulerConfigTable.watcher_active == False)
    await manager_pgvector.execute(query)
    logger.info(f"{invert(f' {job_id} ')} {bold('스케줄러 설정을 초기화하였습니다.')}")


async def start_notice_watcher():
    """
    공지사항 변경 감시를 시작합니다.
    """
    job_id = "start_notice_watcher"
    await init_scheduler_config()
    interval_trigger = IntervalTrigger(seconds=30)  # 10초마다 실행 (config로 변경 가능)
    scheduler.add_job(
        check_new_data,
        interval_trigger,
        id="notice_watcher",
        replace_existing=True,
    )
    scheduler.start()
    logger.info(f"{invert(f' {job_id} ')} {bold('공지사항 변경 감시를 시작합니다.')}")
    
    # 추가된 스케줄러 작업 확인
    jobs = scheduler.get_jobs()
    for job in jobs:
        logger.info(f"등록된 작업: {job}")


def stop_notice_watcher(): 
    """
    공지사항 변경 감시를 중지합니다.
    """
    job_id = "stop_notice_watcher"
    logger.info(f"{invert(f' {job_id} ')} {bold('공지사항 변경 감시를 중지합니다.')}")
    stop_event.set()
    scheduler.remove_all_jobs()
    scheduler.shutdown()


# Main Entry Point
async def start():
    """
    스케줄러 서비스를 시작합니다.
    """
    job_id = "start"

    logger.info(f"{invert(f' {job_id} ')} {bold('스케줄러 서비스를 시작합니다.')}")
    stop_event.clear()

    await start_notice_watcher()

    # 이벤트 루프가 계속 실행되도록 유지
    await stop_event.wait()


def stop():
    """
    스케줄러 서비스를 중지합니다.
    """
    job_id = "stop"
    logger.info(f"{invert(f' {job_id} ')} {bold('스케줄러 서비스를 중지합니다.')}")
    stop_notice_watcher()

