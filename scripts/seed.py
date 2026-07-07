"""
Seed script: populates the database with realistic sample data.
Idempotent - running twice does not create duplicates.
"""

import os
import random
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.config.settings import EngagementState, PrivacyMode, UserRole
from src.models.course import Course
from src.models.engagement_log import EngagementLog
from src.models.session import Session as SessionModel
from src.models.user import User

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/engageiq_dev")
engine = create_engine(DATABASE_URL)

STATES = [
    EngagementState.ENGAGED,
    EngagementState.PASSIVE,
    EngagementState.DISTRACTED,
    EngagementState.DROWSY,
    EngagementState.CONFUSED,
]


def seed():
    with Session(engine) as db:
        # Idempotent check
        if db.query(User).count() > 0:
            print("Database already seeded. Skipping.")
            return

        # 2 Teachers
        teachers = []
        for i in range(1, 3):
            t = User(
                name=f"Teacher {i}",
                email=f"teacher{i}@engageiq.com",
                role=UserRole.TEACHER,
                privacy_mode=PrivacyMode.LOCAL_ONLY,
            )
            db.add(t)
            teachers.append(t)

        # 10 Students
        students = []
        for i in range(1, 11):
            s = User(
                name=f"Student {i}",
                email=f"student{i}@engageiq.com",
                role=UserRole.STUDENT,
                privacy_mode=PrivacyMode.LOCAL_ONLY,
            )
            db.add(s)
            students.append(s)

        db.flush()

        # 3 Courses
        courses = []
        course_data = [
            ("Machine Learning", "ML101"),
            ("Data Structures", "DS201"),
            ("Web Development", "WD301"),
        ]
        for i, (name, code) in enumerate(course_data):
            c = Course(
                name=name,
                code=code,
                teacher_id=teachers[i % 2].id,
            )
            db.add(c)
            courses.append(c)

        db.flush()

        # 5 Sessions
        sessions = []
        for i in range(5):
            sess = SessionModel(
                course_id=courses[i % 3].id,
                start_time=datetime.now() - timedelta(hours=i + 1),
                end_time=datetime.now() - timedelta(hours=i),
                status="completed",
            )
            db.add(sess)
            sessions.append(sess)

        db.flush()

        # 100 Engagement logs
        for i in range(100):
            log = EngagementLog(
                session_id=sessions[i % 5].id,
                user_id=students[i % 10].id,
                score=round(random.uniform(0.3, 1.0), 2),
                state=random.choice(STATES),
                timestamp=datetime.now() - timedelta(minutes=i),
            )
            db.add(log)

        db.commit()
        print(
            "Created 2 teachers, 10 students, 3 courses, 5 sessions, 100 engagement logs"
        )


if __name__ == "__main__":
    seed()
