from pathlib import Path
import subprocess

def get_version():
    try: # Github project
        return subprocess.check_output(['git', 'log', '-n', '1', '--pretty=tformat:%H-%ad', '--date=short'], cwd=Path(__file__).resolve().parent).strip().decode()
    except: # Package
        import pkg_resources
        try:
            version = pkg_resources.get_distribution('cpt').version
            return version
        except pkg_resources.DistributionNotFound:
            return None

def check_version(database):
    print('Version used for interpretation :',database.version.unique().item())
    print('Version using at the moment     :',get_version())
    return 