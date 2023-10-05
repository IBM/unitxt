# Contributing Guide








# Repo principles:

## Git

### Pull
Always pull --rebase before pushing:

    ```bash
    git pull --rebase
    ```
If you have local changes you want to stash:

    ```bash
    git pull --rebase --autostash
    ```

### Commit
Always commit with a [good commit message](https://cbea.ms/git-commit/) and sign off:

Example:

    ```bash
    git commit -s
    ```
### Push
Push into a new branch and open a PR.

Example:

        ```bash
        git push origin main:<my-new-branch-name>
        ```


## Structre

### Layout
The layout of the repo is [src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
