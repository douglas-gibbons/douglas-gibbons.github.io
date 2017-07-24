---
layout: post
title:  "Ansible Playbooks and Environments"
date:   2015-12-04 20:00:00 -0700
categories: [ "Linux" ]
---

Ansible playbooks are wonderfully flexible. If you have been using them a while, you’ve probably looked through the [best practices](http://docs.ansible.com/ansible/playbooks_best_practices.html), but this doesn’t really explain how to set up environment variables for deploying to multiple environments.

Luckily there’s a pretty simple way.

First create an inventory file for your environment, and use it to put all of your environment’s host groups into a special group. For example, here’s one for a staging environment, where we classify all host groups as children of the “staging” group:

```
[web]
web1.staging.zenly.xyz

[database]
database1.staging.zenly.xyz

[staging:children]
web
database
```

You could even create one for using the [packer ansible local provisioner](https://www.packer.io/docs/provisioners/ansible-local.html) if you’re so inclined:

```
[web]
localhost

[database]
localhost

[staging:children]
web
database
```

Now all the staging environment hosts are in the staging group so Ansible will look for group variables in the group_vars/staging file, or in all files in a directory of that name.

Here’s an example showing that full directory structure, with two roles, two playbooks (**web.yml** and **database.yml**), and environment variables in their own groups.  There are even two files for each environment (**main** and **secrets**). Perhaps **“secrets”** holds [ansible-vault](http://docs.ansible.com/ansible/playbooks_vault.html) encrypted secrets for each environment.  There are also inventory files as described above, such as **staging_hosts** for the staging environment and **packer_staging_hosts** for running packer to create images suitable for the staging environment.

```
ansible/
  roles/
    web/
      handlers/
        main.yml
      tasks/
        main.yaml
    database/
      handlers/
        main.yml
      tasks/
        main.yaml
  group_vars/
    staging/
      main
      secrets
    production/
      main
      secrets
  
  staging_hosts
  production_hosts
  packer_staging_hosts
  packer_production_hosts
  web.yml
  database.yml
```

I hope that explains how to use Ansible playbooks with multiple environments.  Enjoy!
