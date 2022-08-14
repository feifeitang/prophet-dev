.PHONY: venv-activate
venv-activate:
	@source env/bin/activate

.PHONY: freeze-reqs
freeze-reqs:
	@python -m pip freeze > requirements.txt

.PHONY: sls-deploy
sls-deploy:
	@sls deploy --region us-west-2

.PHONY: sls-invoke
sls-invoke:
	@sls invoke -f rateHandler

.PHONY: sls-deploy-invoke
sls-deploy-invoke: sls-deploy sls-invoke