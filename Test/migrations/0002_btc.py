# Generated by Django 5.1.4 on 2024-12-16 09:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("Test", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="Btc",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("close", models.IntegerField(blank=True, verbose_name="종가")),
                ("open", models.IntegerField(blank=True, verbose_name="시가")),
                ("low", models.IntegerField(blank=True, verbose_name="저가")),
                ("high", models.IntegerField(blank=True, verbose_name="고가")),
            ],
        ),
    ]
