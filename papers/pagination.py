from django.core.paginator import Paginator, EmptyPage
from .models import Essay

def fetch(page, target_class=None, target_cluster=None, keyword=None, per_page=16):
    """
    根据给定的 essayId 和页码，分页返回同类别的论文
    """
    try:
        essays = Essay.objects.all().order_by('id')

        if target_class is not None:
            target_essay = Essay.objects.get(id=target_class)
            target_category = target_essay.category
            essays = essays.filter(category=target_category).exclude(id=target_class).order_by('id')

        if target_cluster is not None:
            target_essay = Essay.objects.get(id=target_cluster)
            target_category = target_essay.cluster_label
            essays = essays.filter(cluster_label=target_category).exclude(id=target_class).order_by('id')

        if keyword is not None:
            essays = essays.filter(title__icontains=keyword)


        # 分页处理，每页 `per_page` 篇论文
        paginator = Paginator(essays, per_page)
        try:
            paginated_essays = paginator.page(page)
        except EmptyPage:
            paginated_essays = paginator.page(paginator.num_pages)  # 返回最后一页


        # 返回结果
        return {
            'essays': [{'id': essay.id, 'title': essay.title, 'year': essay.year} for essay in paginated_essays],
            'current_page': paginated_essays.number,
            'total_pages': paginator.num_pages,
            # 'total_items': paginator.count
        }
    except Essay.DoesNotExist:
        return {'error': f'Essay with id {target_class} does not exist'}
